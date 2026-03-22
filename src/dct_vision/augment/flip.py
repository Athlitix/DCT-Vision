"""Horizontal and vertical flip in DCT domain.

Horizontal flip: reverse block order along columns, then negate
odd-indexed horizontal frequency coefficients within each block.

Vertical flip: reverse block order along rows, then negate
odd-indexed vertical frequency coefficients within each block.

This avoids any IDCT/DCT -- pure coefficient manipulation.
"""

from __future__ import annotations

import numpy as np

from dct_vision.core.dct_image import DCTImage
from dct_vision.utils.constants import BLOCK_SIZE


def _flip_coeffs_horizontal(coeffs: np.ndarray) -> np.ndarray:
    """Flip coefficients horizontally (left-right)."""
    # Step 1: reverse block column order
    flipped = coeffs[:, ::-1, :, :].copy()
    # Step 2: negate odd horizontal frequency indices (columns in the 8x8 block)
    flipped[:, :, :, 1::2] = -flipped[:, :, :, 1::2]
    return flipped


def _flip_coeffs_vertical(coeffs: np.ndarray) -> np.ndarray:
    """Flip coefficients vertically (top-bottom)."""
    # Step 1: reverse block row order
    flipped = coeffs[::-1, :, :, :].copy()
    # Step 2: negate odd vertical frequency indices (rows in the 8x8 block)
    flipped[:, :, 1::2, :] = -flipped[:, :, 1::2, :]
    return flipped


def horizontal_flip(img: DCTImage) -> DCTImage:
    """Flip image horizontally in DCT domain.

    Parameters
    ----------
    img : DCTImage
        Input image.

    Returns
    -------
    DCTImage
        Horizontally flipped image.
    """
    y = _flip_coeffs_horizontal(img.y_coeffs)
    cb = _flip_coeffs_horizontal(img.cb_coeffs) if img.cb_coeffs is not None else None
    cr = _flip_coeffs_horizontal(img.cr_coeffs) if img.cr_coeffs is not None else None

    return DCTImage(
        y_coeffs=y, cb_coeffs=cb, cr_coeffs=cr,
        quant_tables=img.quant_tables,
        width=img.width, height=img.height,
        comp_info=img.comp_info,
        source_path=img._source_path,
    )


def vertical_flip(img: DCTImage) -> DCTImage:
    """Flip image vertically in DCT domain.

    Parameters
    ----------
    img : DCTImage
        Input image.

    Returns
    -------
    DCTImage
        Vertically flipped image.
    """
    y = _flip_coeffs_vertical(img.y_coeffs)
    cb = _flip_coeffs_vertical(img.cb_coeffs) if img.cb_coeffs is not None else None
    cr = _flip_coeffs_vertical(img.cr_coeffs) if img.cr_coeffs is not None else None

    return DCTImage(
        y_coeffs=y, cb_coeffs=cb, cr_coeffs=cr,
        quant_tables=img.quant_tables,
        width=img.width, height=img.height,
        comp_info=img.comp_info,
        source_path=img._source_path,
    )
