"""Brightness and contrast adjustment in DCT domain."""

from __future__ import annotations

import numpy as np
from dct_vision.core.dct_image import DCTImage
from dct_vision.utils.constants import BLOCK_SIZE


def _dc_offset_for_brightness(offset: float) -> float:
    """Convert pixel-space brightness offset to DC coefficient offset.

    The DC coefficient in an orthonormal 8x8 DCT equals the block mean
    multiplied by sqrt(N*M) where N=M=8. So DC = mean * 8.
    To shift all pixels by `offset`, shift DC by offset * 8 / quant_value.
    We apply the raw scaled offset and let quantization handle the rest.
    """
    # For orthonormal DCT: DC = sum(pixels) / sqrt(64) = mean * 8
    return offset * (BLOCK_SIZE / np.sqrt(BLOCK_SIZE * BLOCK_SIZE))


def adjust_brightness(img: DCTImage, offset: float) -> DCTImage:
    """Adjust image brightness by modifying DC coefficients.

    Parameters
    ----------
    img : DCTImage
        Input image.
    offset : float
        Brightness offset in pixel-value scale (e.g., 30 = brighter).

    Returns
    -------
    DCTImage
        New image with adjusted brightness.
    """
    if offset == 0:
        return DCTImage(
            y_coeffs=img.y_coeffs.copy(),
            cb_coeffs=img.cb_coeffs.copy() if img.cb_coeffs is not None else None,
            cr_coeffs=img.cr_coeffs.copy() if img.cr_coeffs is not None else None,
            quant_tables=img.quant_tables,
            width=img.width,
            height=img.height,
            comp_info=img.comp_info,
        )

    y_coeffs = img.y_coeffs.copy()
    # DC coefficient is at position [0, 0] in each 8x8 block
    # Convert pixel offset to quantized DC offset
    dc_delta = _dc_offset_for_brightness(offset)

    # Account for quantization: the stored coefficient is quantized,
    # so we need to scale by the quantization table value
    qtable_idx = 0
    if img.comp_info:
        qtable_idx = img.comp_info[0].get("quant_tbl_no", 0)
    qtable = img.quant_tables[min(qtable_idx, len(img.quant_tables) - 1)]
    dc_quant = qtable[0, 0]

    # Add the offset in quantized coefficient space
    dc_offset_quantized = int(round(dc_delta / dc_quant))
    y_coeffs[:, :, 0, 0] = y_coeffs[:, :, 0, 0] + dc_offset_quantized

    return DCTImage(
        y_coeffs=y_coeffs,
        cb_coeffs=img.cb_coeffs.copy() if img.cb_coeffs is not None else None,
        cr_coeffs=img.cr_coeffs.copy() if img.cr_coeffs is not None else None,
        quant_tables=img.quant_tables,
        width=img.width,
        height=img.height,
        comp_info=img.comp_info,
    )


def adjust_contrast(img: DCTImage, factor: float) -> DCTImage:
    """Adjust image contrast by scaling AC coefficients.

    DC coefficient (block mean) is preserved. AC coefficients (deviation
    from mean) are scaled by the given factor.

    Parameters
    ----------
    img : DCTImage
        Input image.
    factor : float
        Contrast factor. >1 increases contrast, <1 decreases. Must be >= 0.

    Returns
    -------
    DCTImage
        New image with adjusted contrast.

    Raises
    ------
    ValueError
        If factor < 0.
    """
    if factor < 0:
        raise ValueError(f"Contrast factor must be >= 0, got {factor}")

    if factor == 1.0:
        return DCTImage(
            y_coeffs=img.y_coeffs.copy(),
            cb_coeffs=img.cb_coeffs.copy() if img.cb_coeffs is not None else None,
            cr_coeffs=img.cr_coeffs.copy() if img.cr_coeffs is not None else None,
            quant_tables=img.quant_tables,
            width=img.width,
            height=img.height,
            comp_info=img.comp_info,
        )

    y_coeffs = img.y_coeffs.astype(np.float32)

    # Preserve DC, scale AC
    dc = y_coeffs[:, :, 0, 0].copy()
    y_coeffs *= factor
    y_coeffs[:, :, 0, 0] = dc

    return DCTImage(
        y_coeffs=np.round(y_coeffs).astype(np.int16),
        cb_coeffs=img.cb_coeffs.copy() if img.cb_coeffs is not None else None,
        cr_coeffs=img.cr_coeffs.copy() if img.cr_coeffs is not None else None,
        quant_tables=img.quant_tables,
        width=img.width,
        height=img.height,
        comp_info=img.comp_info,
    )
