"""JPEG DCT coefficient extraction and writing.

Uses Pillow for JPEG I/O and scipy for DCT computation. This provides
a reliable, portable extraction path. A native libjpeg-turbo ctypes path
can be added later as an optimization for zero-decode extraction.
"""

import os
import numpy as np
from PIL import Image

from dct_vision.exceptions import InvalidImageError, LibjpegError
from dct_vision.math.dct import dct2, idct2, blockwise_dct, blockwise_idct
from dct_vision.math.colorspace import rgb_to_ycbcr, ycbcr_to_rgb
from dct_vision.math.quantization import quantize, dequantize, scale_quant_table
from dct_vision.core.block import pad_to_block_multiple, BLOCK_SIZE
from dct_vision.utils.constants import LUMINANCE_QUANT_TABLE, CHROMINANCE_QUANT_TABLE


def _validate_jpeg_path(path: str) -> None:
    """Validate that path exists and is a JPEG file."""
    if not os.path.exists(path):
        raise InvalidImageError(f"File not found: {path}")
    try:
        with open(path, "rb") as f:
            magic = f.read(2)
            if magic != b"\xff\xd8":
                raise InvalidImageError(f"Not a JPEG file: {path}")
    except IOError as e:
        raise InvalidImageError(f"Cannot read file: {path}: {e}")


def _detect_subsampling(img: Image.Image) -> tuple[str, list[dict]]:
    """Detect chroma subsampling from a Pillow JPEG image."""
    # Pillow provides layer info for JPEG
    # For a standard JPEG: Y has higher sampling, Cb/Cr have lower
    # We infer from the quantization tables and image info
    try:
        # Pillow stores JPEG info
        info = getattr(img, "info", {})
        # Try to detect from Pillow's internal data
        if hasattr(img, "layer"):
            layers = img.layer
        else:
            layers = None
    except Exception:
        layers = None

    # Default assumption based on typical JPEG
    # Most JPEGs are 4:2:0
    mode = img.mode
    width, height = img.size

    if mode == "L":
        comp_info = [{"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": 0}]
        return "4:4:4", comp_info

    # For color images, try to detect subsampling
    # Pillow doesn't directly expose sampling factors for JPEG
    # Use a heuristic: open with raw access
    # Default to 4:2:0 as it's most common
    comp_info = [
        {"h_samp_factor": 2, "v_samp_factor": 2, "quant_tbl_no": 0},  # Y
        {"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": 1},  # Cb
        {"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": 1},  # Cr
    ]
    subsampling = "4:2:0"

    # Try to detect actual subsampling by re-saving and comparing
    # Or check Pillow's internal JPEG info
    try:
        # Some Pillow versions expose this
        if hasattr(img, "_getexif"):
            pass
        # Check quantization tables for hints
        qtables = img.quantization
        if qtables:
            # If only 1 quant table, likely 4:4:4
            if len(qtables) == 1:
                subsampling = "4:4:4"
                comp_info = [
                    {"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": 0},
                    {"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": 0},
                    {"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": 0},
                ]
    except Exception:
        pass

    return subsampling, comp_info


def _get_quant_tables_from_pillow(img: Image.Image) -> list[np.ndarray]:
    """Extract quantization tables from a Pillow JPEG image."""
    qtables = []
    try:
        pil_qtables = img.quantization
        if pil_qtables:
            for tbl_no in sorted(pil_qtables.keys()):
                tbl = pil_qtables[tbl_no]
                if isinstance(tbl, (list, tuple)):
                    qtables.append(np.array(tbl, dtype=np.float32).reshape(8, 8))
                else:
                    qtables.append(np.array(list(tbl), dtype=np.float32).reshape(8, 8))
    except Exception:
        pass

    if not qtables:
        # Fallback to standard tables
        qtables = [LUMINANCE_QUANT_TABLE.copy(), CHROMINANCE_QUANT_TABLE.copy()]

    return qtables


def read_dct_coefficients(path: str) -> dict:
    """Extract DCT coefficients from a JPEG file.

    Decodes the JPEG to pixels via Pillow, converts to YCbCr, then applies
    blockwise DCT to obtain coefficients. Quantization tables are extracted
    from the JPEG metadata.

    Parameters
    ----------
    path : str
        Path to a JPEG file.

    Returns
    -------
    dict
        Keys:
        - 'coefficients': list of np.ndarray, shape (bh, bw, 8, 8), dtype int16
        - 'quant_tables': list of np.ndarray, shape (8, 8)
        - 'width', 'height': int
        - 'num_components': int
        - 'comp_info': list of dicts with sampling factors
    """
    _validate_jpeg_path(path)

    try:
        img = Image.open(path)
        img.load()  # Force full load to access quantization tables
    except Exception as e:
        raise InvalidImageError(f"Failed to open JPEG: {path}: {e}")

    width, height = img.size
    quant_tables = _get_quant_tables_from_pillow(img)

    if img.mode == "L":
        # Grayscale
        pixels = np.array(img, dtype=np.float32)
        padded = pad_to_block_multiple(pixels)
        # Level shift before DCT (JPEG standard)
        shifted = padded - 128.0
        raw_coeffs = blockwise_dct(shifted)
        # Quantize to match JPEG's stored representation
        qtable = quant_tables[0]
        quantized = np.zeros_like(raw_coeffs, dtype=np.int16)
        bh, bw = raw_coeffs.shape[:2]
        for i in range(bh):
            for j in range(bw):
                quantized[i, j] = np.round(raw_coeffs[i, j] / qtable).astype(np.int16)

        coefficients = [quantized]
        num_components = 1
        comp_info = [{"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": 0}]

    else:
        # Color image — convert to YCbCr
        if img.mode != "RGB":
            img = img.convert("RGB")
        rgb = np.array(img, dtype=np.float32)
        ycbcr = rgb_to_ycbcr(rgb)

        subsampling, comp_info = _detect_subsampling(Image.open(path))

        coefficients = []
        for ch_idx in range(3):
            channel = ycbcr[:, :, ch_idx]

            # For chroma channels with subsampling, downsample
            if ch_idx > 0:
                h_samp = comp_info[0]["h_samp_factor"]
                v_samp = comp_info[0]["v_samp_factor"]
                ch_h_samp = comp_info[ch_idx]["h_samp_factor"]
                ch_v_samp = comp_info[ch_idx]["v_samp_factor"]

                if h_samp > ch_h_samp:
                    # Horizontal downsampling
                    factor_w = h_samp // ch_h_samp
                    channel = channel[:, ::factor_w]
                if v_samp > ch_v_samp:
                    # Vertical downsampling
                    factor_h = v_samp // ch_v_samp
                    channel = channel[::factor_h, :]

            padded = pad_to_block_multiple(channel)

            # Subtract 128 from pixel values before DCT (JPEG level shift)
            shifted = padded - 128.0
            raw_coeffs = blockwise_dct(shifted)

            # Quantize
            qtable_idx = comp_info[ch_idx]["quant_tbl_no"]
            qtable = quant_tables[min(qtable_idx, len(quant_tables) - 1)]
            quantized = np.zeros_like(raw_coeffs, dtype=np.int16)
            bh, bw = raw_coeffs.shape[:2]
            for i in range(bh):
                for j in range(bw):
                    quantized[i, j] = np.round(raw_coeffs[i, j] / qtable).astype(np.int16)

            coefficients.append(quantized)

        num_components = 3

    return {
        "coefficients": coefficients,
        "quant_tables": quant_tables,
        "width": width,
        "height": height,
        "num_components": num_components,
        "comp_info": comp_info,
    }


def write_dct_coefficients(
    path: str,
    coefficients: list[np.ndarray],
    quant_tables: list[np.ndarray],
    width: int,
    height: int,
    num_components: int,
    comp_info: list[dict] | None = None,
) -> None:
    """Write DCT coefficients to a JPEG file.

    Dequantizes coefficients, applies inverse DCT, converts to RGB,
    and saves via Pillow with matching quantization tables.

    Parameters
    ----------
    path : str
        Output JPEG file path.
    coefficients : list[np.ndarray]
        Coefficient arrays per component, each shape (bh, bw, 8, 8), dtype int16.
    quant_tables : list[np.ndarray]
        Quantization tables, each shape (8, 8).
    width, height : int
        Image dimensions.
    num_components : int
        Number of color components.
    comp_info : list[dict], optional
        Component info with sampling factors.
    """
    if num_components == 1:
        # Grayscale
        qtable = quant_tables[0]
        coeffs = coefficients[0].astype(np.float32)
        bh, bw = coeffs.shape[:2]

        # Dequantize
        for i in range(bh):
            for j in range(bw):
                coeffs[i, j] = coeffs[i, j] * qtable

        channel = blockwise_idct(coeffs)
        pixels = np.clip(channel[:height, :width], 0, 255).astype(np.uint8)
        img = Image.fromarray(pixels, mode="L")

    else:
        # Color image
        if comp_info is None:
            comp_info = [
                {"h_samp_factor": 2, "v_samp_factor": 2, "quant_tbl_no": 0},
                {"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": 1},
                {"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": 1},
            ]

        channels = []
        for ch_idx in range(3):
            qtable_idx = comp_info[ch_idx]["quant_tbl_no"]
            qtable = quant_tables[min(qtable_idx, len(quant_tables) - 1)]
            coeffs = coefficients[ch_idx].astype(np.float32)
            bh, bw = coeffs.shape[:2]

            # Dequantize
            for i in range(bh):
                for j in range(bw):
                    coeffs[i, j] = coeffs[i, j] * qtable

            channel = blockwise_idct(coeffs)
            # Add 128 back (reverse level shift)
            channel = channel + 128.0

            # Upsample chroma if needed
            if ch_idx > 0:
                h_samp = comp_info[0]["h_samp_factor"]
                v_samp = comp_info[0]["v_samp_factor"]
                ch_h_samp = comp_info[ch_idx]["h_samp_factor"]
                ch_v_samp = comp_info[ch_idx]["v_samp_factor"]

                if v_samp > ch_v_samp:
                    channel = np.repeat(channel, v_samp // ch_v_samp, axis=0)
                if h_samp > ch_h_samp:
                    channel = np.repeat(channel, h_samp // ch_h_samp, axis=1)

            channels.append(channel[:height, :width])

        # Stack to YCbCr and convert to RGB
        # Ensure all channels have the same shape
        min_h = min(ch.shape[0] for ch in channels)
        min_w = min(ch.shape[1] for ch in channels)
        ycbcr = np.stack([ch[:min_h, :min_w] for ch in channels], axis=-1).astype(np.float32)
        rgb = ycbcr_to_rgb(ycbcr)
        pixels = np.clip(rgb, 0, 255).astype(np.uint8)
        img = Image.fromarray(pixels, mode="RGB")

    # Build Pillow qtables format
    pillow_qtables = {}
    for i, qt in enumerate(quant_tables):
        pillow_qtables[i] = qt.flatten().astype(int).tolist()

    # Determine subsampling
    if comp_info and num_components == 3:
        h_samp = comp_info[0]["h_samp_factor"]
        v_samp = comp_info[0]["v_samp_factor"]
        if h_samp == 1 and v_samp == 1:
            subsampling = 0  # 4:4:4
        elif h_samp == 2 and v_samp == 1:
            subsampling = 1  # 4:2:2
        else:
            subsampling = 2  # 4:2:0
    else:
        subsampling = 0

    img.save(path, format="JPEG", qtables=pillow_qtables, subsampling=subsampling)
