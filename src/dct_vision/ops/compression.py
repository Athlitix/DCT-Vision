"""Compression operations: requantize and coefficient pruning.

Change JPEG quality without pixel decode, or reduce file size
by zeroing small coefficients.
"""

from __future__ import annotations

import numpy as np

from dct_vision.core.dct_image import DCTImage
from dct_vision.math.quantization import scale_quant_table
from dct_vision.utils.constants import LUMINANCE_QUANT_TABLE, CHROMINANCE_QUANT_TABLE


def requantize(img: DCTImage, quality: int) -> DCTImage:
    """Change JPEG quality by requantizing coefficients.

    Dequantizes with the original table, then requantizes with a new
    table derived from the target quality. No pixel decode needed.

    Parameters
    ----------
    img : DCTImage
        Input image.
    quality : int
        Target JPEG quality (1-100).

    Returns
    -------
    DCTImage
        Image with new quantization.

    Raises
    ------
    ValueError
        If quality not in [1, 100].
    """
    if quality < 1 or quality > 100:
        raise ValueError(f"quality must be in [1, 100], got {quality}")

    old_luma_qt = img.quant_tables[0]
    new_luma_qt = scale_quant_table(LUMINANCE_QUANT_TABLE, quality)

    # Dequantize with old table, requantize with new
    y = img.y_coeffs.astype(np.float32) * old_luma_qt
    y = np.round(y / new_luma_qt).astype(np.int16)

    new_quant_tables = [new_luma_qt]

    cb = img.cb_coeffs
    cr = img.cr_coeffs
    if cb is not None:
        old_ch_qt = img.quant_tables[min(1, len(img.quant_tables) - 1)]
        new_ch_qt = scale_quant_table(CHROMINANCE_QUANT_TABLE, quality)

        cb = cb.astype(np.float32) * old_ch_qt
        cb = np.round(cb / new_ch_qt).astype(np.int16)
        cr = cr.astype(np.float32) * old_ch_qt
        cr = np.round(cr / new_ch_qt).astype(np.int16)

        new_quant_tables.append(new_ch_qt)

    return DCTImage(
        y_coeffs=y, cb_coeffs=cb, cr_coeffs=cr,
        quant_tables=new_quant_tables,
        width=img.width, height=img.height,
        comp_info=img.comp_info,
        # No source_path -- quant tables changed, can't use native write
    )


def prune_coefficients(img: DCTImage, threshold: int = 2) -> DCTImage:
    """Zero out small AC coefficients to reduce file size.

    Coefficients with absolute value <= threshold are set to zero.
    DC coefficients are always preserved.

    Parameters
    ----------
    img : DCTImage
        Input image.
    threshold : int
        Coefficients with |value| <= threshold are zeroed. 0 = no pruning.

    Returns
    -------
    DCTImage
    """
    if threshold == 0:
        return DCTImage(
            y_coeffs=img.y_coeffs.copy(),
            cb_coeffs=img.cb_coeffs.copy() if img.cb_coeffs is not None else None,
            cr_coeffs=img.cr_coeffs.copy() if img.cr_coeffs is not None else None,
            quant_tables=img.quant_tables,
            width=img.width, height=img.height,
            comp_info=img.comp_info,
            source_path=img._source_path,
        )

    y = img.y_coeffs.copy()
    dc = y[:, :, 0, 0].copy()
    y[np.abs(y) <= threshold] = 0
    y[:, :, 0, 0] = dc  # restore DC

    cb = img.cb_coeffs
    cr = img.cr_coeffs
    if cb is not None:
        cb = cb.copy()
        cb_dc = cb[:, :, 0, 0].copy()
        cb[np.abs(cb) <= threshold] = 0
        cb[:, :, 0, 0] = cb_dc

        cr = cr.copy()
        cr_dc = cr[:, :, 0, 0].copy()
        cr[np.abs(cr) <= threshold] = 0
        cr[:, :, 0, 0] = cr_dc

    return DCTImage(
        y_coeffs=y, cb_coeffs=cb, cr_coeffs=cr,
        quant_tables=img.quant_tables,
        width=img.width, height=img.height,
        comp_info=img.comp_info,
        source_path=img._source_path,
    )
