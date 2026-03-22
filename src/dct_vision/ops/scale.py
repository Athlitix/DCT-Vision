"""Downscaling via DCT frequency truncation.

Downscaling by factor 2 works by keeping only the lower-frequency
half of coefficients in each block, effectively bandlimiting the image.
We merge adjacent blocks and truncate high frequencies.
"""

from __future__ import annotations
import math

import numpy as np
from scipy.fft import dctn, idctn

from dct_vision.core.dct_image import DCTImage
from dct_vision.utils.constants import BLOCK_SIZE


def _downscale_channel(coeffs: np.ndarray, qtable: np.ndarray, factor: int) -> np.ndarray:
    """Downscale a single channel's DCT coefficients by the given factor.

    Strategy: dequantize blocks -> IDCT to pixels -> subsample -> re-block -> DCT -> quantize.
    This is the simplest correct approach for arbitrary factors.
    """
    bh, bw = coeffs.shape[:2]
    # Reconstruct pixel channel
    h = bh * BLOCK_SIZE
    w = bw * BLOCK_SIZE

    channel = np.zeros((h, w), dtype=np.float32)
    for i in range(bh):
        for j in range(bw):
            block = coeffs[i, j].astype(np.float32) * qtable
            channel[
                i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE,
                j * BLOCK_SIZE : (j + 1) * BLOCK_SIZE,
            ] = idctn(block, type=2, norm="ortho")

    # Downsample the pixel channel
    new_h = h // factor
    new_w = w // factor
    # Simple area-average downsampling
    downsampled = channel.reshape(new_h, factor, new_w, factor).mean(axis=(1, 3))

    # Pad to block multiple
    pad_h = math.ceil(new_h / BLOCK_SIZE) * BLOCK_SIZE
    pad_w = math.ceil(new_w / BLOCK_SIZE) * BLOCK_SIZE
    padded = np.zeros((pad_h, pad_w), dtype=np.float32)
    padded[:new_h, :new_w] = downsampled

    # Re-compute DCT blocks and quantize
    new_bh = pad_h // BLOCK_SIZE
    new_bw = pad_w // BLOCK_SIZE
    new_coeffs = np.zeros((new_bh, new_bw, BLOCK_SIZE, BLOCK_SIZE), dtype=np.int16)

    for i in range(new_bh):
        for j in range(new_bw):
            block = padded[
                i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE,
                j * BLOCK_SIZE : (j + 1) * BLOCK_SIZE,
            ]
            dct_block = dctn(block, type=2, norm="ortho")
            new_coeffs[i, j] = np.round(dct_block / qtable).astype(np.int16)

    return new_coeffs


def downscale(img: DCTImage, factor: int = 2) -> DCTImage:
    """Downscale image by the given factor.

    Parameters
    ----------
    img : DCTImage
        Input image.
    factor : int
        Downscale factor. Must be a power of 2 (2, 4, 8, ...).

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
