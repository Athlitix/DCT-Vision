"""Cross-block boundary strategy for operations spanning multiple blocks.

For operations that need context beyond a single 8x8 block (blur with
large sigma, edge detection at block boundaries), we use an overlapping
3x3 block neighborhood approach:

1. For each block (i, j), gather the 3x3 neighborhood (9 blocks, 24x24 pixels)
2. Dequantize and IDCT only this local 24x24 patch
3. Apply the spatial operation on the full 24x24 patch
4. Extract the center 8x8 result
5. DCT and quantize back

This avoids full-image decode while providing cross-boundary context.

Block caching: when iterating left-to-right, top-to-bottom, previously
decoded blocks are cached and reused. Adjacent neighborhoods share 6 of 9
blocks, reducing redundant IDCT by ~66%.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter

from dct_vision.core.dct_image import DCTImage
from dct_vision.utils.constants import BLOCK_SIZE


def _dequantize_block(coeffs: np.ndarray, qtable: np.ndarray) -> np.ndarray:
    """Dequantize and IDCT a single block to pixels."""
    return idctn(coeffs.astype(np.float32) * qtable, type=2, norm="ortho")


def _build_pixel_cache(coeffs: np.ndarray, qtable: np.ndarray) -> np.ndarray:
    """Batch-IDCT all blocks into a pixel cache.

    Returns shape (bh, bw, 8, 8) of decoded pixel blocks.
    """
    bh, bw = coeffs.shape[:2]
    dequant = coeffs.astype(np.float32) * qtable
    flat = dequant.reshape(-1, BLOCK_SIZE, BLOCK_SIZE)
    pixels_flat = idctn(flat, type=2, norm="ortho", axes=(-2, -1))
    return pixels_flat.reshape(bh, bw, BLOCK_SIZE, BLOCK_SIZE)


def _extract_neighborhood(
    pixel_cache: np.ndarray,
    block_row: int,
    block_col: int,
) -> np.ndarray:
    """Extract 3x3 block neighborhood as a 24x24 pixel patch.

    Handles boundary by zero-padding.
    """
    bh, bw = pixel_cache.shape[:2]
    patch = np.zeros((3 * BLOCK_SIZE, 3 * BLOCK_SIZE), dtype=np.float32)

    for di in range(-1, 2):
        for dj in range(-1, 2):
            ni = block_row + di
            nj = block_col + dj
            if 0 <= ni < bh and 0 <= nj < bw:
                pi = (di + 1) * BLOCK_SIZE
                pj = (dj + 1) * BLOCK_SIZE
                patch[pi : pi + BLOCK_SIZE, pj : pj + BLOCK_SIZE] = pixel_cache[ni, nj]

    return patch


def _process_channel_cross_block(
    coeffs: np.ndarray,
    qtable: np.ndarray,
    operation,
) -> np.ndarray:
    """Apply a spatial operation using 3x3 block neighborhoods.

    Parameters
    ----------
    coeffs : np.ndarray
        DCT coefficients, shape (bh, bw, 8, 8), dtype int16.
    qtable : np.ndarray
        Quantization table, shape (8, 8).
    operation : callable
        Function that takes a 24x24 float32 pixel patch and returns
        a processed 24x24 patch.

    Returns
    -------
    np.ndarray
        New coefficients, shape (bh, bw, 8, 8), dtype int16.
    """
    bh, bw = coeffs.shape[:2]

    # Decode all blocks at once (batch IDCT)
    pixel_cache = _build_pixel_cache(coeffs, qtable)

    # Process each block using its 3x3 neighborhood
    new_coeffs = np.zeros_like(coeffs)

    for i in range(bh):
        for j in range(bw):
            # Get 24x24 neighborhood
            patch = _extract_neighborhood(pixel_cache, i, j)

            # Apply spatial operation on the full neighborhood
            processed = operation(patch)

            # Extract the center 8x8 block
            center = processed[
                BLOCK_SIZE : 2 * BLOCK_SIZE,
                BLOCK_SIZE : 2 * BLOCK_SIZE,
            ]

            # DCT and quantize
            dct_block = dctn(center, type=2, norm="ortho")
            new_coeffs[i, j] = np.round(dct_block / qtable).astype(np.int16)

    return new_coeffs


def cross_block_blur(
    img: DCTImage,
    sigma: float = 3.0,
    channels: str = "all",
) -> DCTImage:
    """Gaussian blur with cross-block boundary handling.

    Uses 3x3 block neighborhood strategy to avoid seam artifacts
    at block boundaries, especially for large sigma values.

    Parameters
    ----------
    img : DCTImage
        Input image.
    sigma : float
        Gaussian blur sigma. Must be > 0.
    channels : str
        'all', 'luma', or 'chroma'.

    Returns
    -------
    DCTImage
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    def blur_op(patch):
        return gaussian_filter(patch, sigma=sigma)

    # Process luma
    if channels in ("all", "luma"):
        qtable_idx = 0
        if img.comp_info:
            qtable_idx = img.comp_info[0].get("quant_tbl_no", 0)
        luma_qt = img.quant_tables[min(qtable_idx, len(img.quant_tables) - 1)]
        y_coeffs = _process_channel_cross_block(img.y_coeffs, luma_qt, blur_op)
    else:
        y_coeffs = img.y_coeffs.copy()

    # Process chroma
    if img.cb_coeffs is not None and channels in ("all", "chroma"):
        ch_qtable_idx = 1 if img.comp_info and len(img.comp_info) > 1 else 0
        if img.comp_info:
            ch_qtable_idx = img.comp_info[1].get("quant_tbl_no", 1)
        ch_qt = img.quant_tables[min(ch_qtable_idx, len(img.quant_tables) - 1)]
        cb_coeffs = _process_channel_cross_block(img.cb_coeffs, ch_qt, blur_op)
        cr_coeffs = _process_channel_cross_block(img.cr_coeffs, ch_qt, blur_op)
    else:
        cb_coeffs = img.cb_coeffs.copy() if img.cb_coeffs is not None else None
        cr_coeffs = img.cr_coeffs.copy() if img.cr_coeffs is not None else None

    return DCTImage(
        y_coeffs=y_coeffs,
        cb_coeffs=cb_coeffs,
        cr_coeffs=cr_coeffs,
        quant_tables=img.quant_tables,
        width=img.width,
        height=img.height,
        comp_info=img.comp_info,
        source_path=img._source_path,
    )


def cross_block_edges(
    img: DCTImage,
    method: str = "laplacian",
) -> DCTImage:
    """Edge detection with cross-block boundary handling.

    Uses 3x3 block neighborhoods so edges at block boundaries
    are properly detected.

    Parameters
    ----------
    img : DCTImage
        Input image.
    method : str
        'laplacian' (default).

    Returns
    -------
    DCTImage
        Grayscale edge map.
    """
    from scipy.ndimage import laplace

    def edge_op(patch):
        return np.abs(laplace(patch))

    qtable_idx = 0
    if img.comp_info:
        qtable_idx = img.comp_info[0].get("quant_tbl_no", 0)
    luma_qt = img.quant_tables[min(qtable_idx, len(img.quant_tables) - 1)]

    y_coeffs = _process_channel_cross_block(img.y_coeffs, luma_qt, edge_op)

    return DCTImage(
        y_coeffs=y_coeffs,
        cb_coeffs=None,
        cr_coeffs=None,
        quant_tables=[luma_qt],
        width=img.width,
        height=img.height,
        comp_info=[{"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": 0}],
        source_path=img._source_path,
    )
