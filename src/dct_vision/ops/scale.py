"""Downscaling via DCT-domain block merging.

For 2x downscale: group 2x2 adjacent DCT blocks and compute the output
8x8 DCT block using a precomputed transform matrix. This avoids any
IDCT/DCT calls entirely -- pure matrix multiplication in frequency domain.

The transform matrix T maps four 8x8 coefficient blocks to one 8x8 block:
    output = T @ [block_TL, block_TR, block_BL, block_BR]

T is derived from the relationship between the 16x16 DCT (composed from
four 8x8 DCTs) and the 8x8 DCT of the area-averaged 8x8 result.
"""

from __future__ import annotations
from functools import lru_cache

import numpy as np
from scipy.fft import dctn, idctn

from dct_vision.core.dct_image import DCTImage
from dct_vision.utils.constants import BLOCK_SIZE


@lru_cache(maxsize=1)
def _build_downscale_2x_matrix() -> np.ndarray:
    """Build the transform matrix that maps 2x2 block groups to one block.

    Precomputes by solving: for each of the 64 output DCT basis patterns,
    what linear combination of the 4x64=256 input DCT coefficients produces it?

    Returns shape (64, 4, 64) reshaped for batch application.
    Actually returns (8, 8, 4, 8, 8) for direct broadcasting.
    """
    N = BLOCK_SIZE  # 8

    # Build the transform by running all 256 unit basis vectors through
    # IDCT -> assemble 16x16 -> downsample -> DCT
    # This gives us the exact linear mapping.

    # For each of the 4 input blocks and each of the 64 coefficient positions,
    # set that coefficient to 1, IDCT, assemble, downsample, DCT.
    # Result: T[out_u, out_v, block_idx, in_u, in_v] = weight

    T = np.zeros((N, N, 4, N, N), dtype=np.float32)

    for block_idx in range(4):
        br = block_idx // 2  # 0 or 1 (top/bottom)
        bc = block_idx % 2   # 0 or 1 (left/right)

        for u in range(N):
            for v in range(N):
                # Set one coefficient to 1
                inp = np.zeros((N, N), dtype=np.float32)
                inp[u, v] = 1.0

                # IDCT to pixels
                pixels = idctn(inp, type=2, norm="ortho")

                # Place into 16x16 patch at the right position
                patch = np.zeros((2 * N, 2 * N), dtype=np.float32)
                patch[br * N:(br + 1) * N, bc * N:(bc + 1) * N] = pixels

                # Area-average downsample 16x16 -> 8x8
                downsampled = patch.reshape(N, 2, N, 2).mean(axis=(1, 3))

                # DCT the result
                out_coeffs = dctn(downsampled, type=2, norm="ortho")

                T[:, :, block_idx, u, v] = out_coeffs

    return T


def _downscale_channel_2x(coeffs: np.ndarray, qtable: np.ndarray) -> np.ndarray:
    """Downscale a channel by factor 2 using precomputed transform matrix.

    Pure frequency-domain operation -- no IDCT or DCT calls at runtime.
    """
    bh, bw = coeffs.shape[:2]

    # Ensure even block dimensions
    if bh % 2 != 0 or bw % 2 != 0:
        new_bh = bh + (bh % 2)
        new_bw = bw + (bw % 2)
        padded = np.zeros((new_bh, new_bw, BLOCK_SIZE, BLOCK_SIZE), dtype=coeffs.dtype)
        padded[:bh, :bw] = coeffs
        coeffs = padded
        bh, bw = new_bh, new_bw

    # Dequantize
    dequant = coeffs.astype(np.float32) * qtable  # (bh, bw, 8, 8)

    gh = bh // 2
    gw = bw // 2

    # Get transform matrix: (8, 8, 4, 8, 8)
    T = _build_downscale_2x_matrix()

    # Gather 2x2 block groups into (gh, gw, 4, 8, 8)
    groups = np.zeros((gh, gw, 4, BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)
    groups[:, :, 0] = dequant[0::2, 0::2]   # top-left
    groups[:, :, 1] = dequant[0::2, 1::2]   # top-right
    groups[:, :, 2] = dequant[1::2, 0::2]   # bottom-left
    groups[:, :, 3] = dequant[1::2, 1::2]   # bottom-right

    # Apply transform: T has shape (8, 8, 4, 8, 8), groups has (gh, gw, 4, 8, 8)
    # output[g, h, ou, ov] = sum over (b, iu, iv) of T[ou, ov, b, iu, iv] * groups[g, h, b, iu, iv]
    result = np.einsum("uvbij,ghbij->ghuv", T, groups)

    # Requantize
    new_coeffs = np.round(result / qtable).astype(np.int16)

    return new_coeffs


def _downscale_channel(coeffs: np.ndarray, qtable: np.ndarray, factor: int) -> np.ndarray:
    """Downscale a channel by the given power-of-2 factor."""
    result = coeffs
    current_factor = factor
    while current_factor > 1:
        result = _downscale_channel_2x(result, qtable)
        current_factor //= 2
    return result


def downscale(img: DCTImage, factor: int = 2) -> DCTImage:
    """Downscale image by the given factor.

    Uses a precomputed DCT-domain transform matrix to merge 2x2 block
    groups into single blocks. No IDCT/DCT calls at runtime -- pure
    matrix multiplication in frequency domain.

    Parameters
    ----------
    img : DCTImage
        Input image.
    factor : int
        Downscale factor. Must be a positive power of 2 (2, 4, 8, ...).

    Returns
    -------
    DCTImage
        Downscaled image.

    Raises
    ------
    ValueError
        If factor is not a positive power of 2.
    """
    if factor < 1 or (factor & (factor - 1)) != 0:
        raise ValueError(f"factor must be a positive power of 2, got {factor}")

    if factor == 1:
        return DCTImage(
            y_coeffs=img.y_coeffs.copy(),
            cb_coeffs=img.cb_coeffs.copy() if img.cb_coeffs is not None else None,
            cr_coeffs=img.cr_coeffs.copy() if img.cr_coeffs is not None else None,
            quant_tables=img.quant_tables,
            width=img.width,
            height=img.height,
            comp_info=img.comp_info,
        )

    new_width = img.width // factor
    new_height = img.height // factor

    # Downscale luma
    qtable_idx = 0
    if img.comp_info:
        qtable_idx = img.comp_info[0].get("quant_tbl_no", 0)
    luma_qtable = img.quant_tables[min(qtable_idx, len(img.quant_tables) - 1)]
    y_coeffs = _downscale_channel(img.y_coeffs, luma_qtable, factor)

    # Downscale chroma
    cb_coeffs = None
    cr_coeffs = None
    if img.cb_coeffs is not None:
        chroma_qtable_idx = 1 if img.comp_info and len(img.comp_info) > 1 else 0
        if img.comp_info:
            chroma_qtable_idx = img.comp_info[1].get("quant_tbl_no", 1)
        chroma_qtable = img.quant_tables[min(chroma_qtable_idx, len(img.quant_tables) - 1)]
        cb_coeffs = _downscale_channel(img.cb_coeffs, chroma_qtable, factor)
        cr_coeffs = _downscale_channel(img.cr_coeffs, chroma_qtable, factor)

    return DCTImage(
        y_coeffs=y_coeffs,
        cb_coeffs=cb_coeffs,
        cr_coeffs=cr_coeffs,
        quant_tables=img.quant_tables,
        width=new_width,
        height=new_height,
        comp_info=img.comp_info,
    )


def upscale(img: DCTImage, factor: int = 2, quality: int = 85) -> DCTImage:
    """Upscale image by the given factor.

    Upscaling requires pixel interpolation, which has no efficient
    DCT-domain equivalent. This convenience wrapper decodes to pixels,
    upscales via Pillow (LANCZOS), and re-encodes to DCTImage.

    Parameters
    ----------
    img : DCTImage
        Input image.
    factor : int
        Upscale factor. Must be a positive power of 2 (2, 4, 8, ...).
    quality : int
        JPEG quality for re-encoding (default 85).

    Returns
    -------
    DCTImage
        Upscaled image.

    Raises
    ------
    ValueError
        If factor is not a positive power of 2.
    """
    from PIL import Image

    if factor < 1 or (factor & (factor - 1)) != 0:
        raise ValueError(f"factor must be a positive power of 2, got {factor}")

    if factor == 1:
        return DCTImage(
            y_coeffs=img.y_coeffs.copy(),
            cb_coeffs=img.cb_coeffs.copy() if img.cb_coeffs is not None else None,
            cr_coeffs=img.cr_coeffs.copy() if img.cr_coeffs is not None else None,
            quant_tables=img.quant_tables,
            width=img.width,
            height=img.height,
            comp_info=img.comp_info,
        )

    pixels = img.to_pixels()
    new_w = img.width * factor
    new_h = img.height * factor

    if pixels.ndim == 2:
        pil_img = Image.fromarray(pixels, mode="L")
    else:
        pil_img = Image.fromarray(pixels, mode="RGB")

    upscaled = pil_img.resize((new_w, new_h), Image.LANCZOS)
    return DCTImage.from_array(np.array(upscaled, dtype=np.uint8), quality=quality)
