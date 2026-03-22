"""Brightness and contrast jitter for data augmentation.

Random perturbations applied directly in DCT domain.
"""

from __future__ import annotations

import numpy as np

from dct_vision.core.dct_image import DCTImage
from dct_vision.ops.color import adjust_brightness, adjust_contrast


def brightness_jitter(
    img: DCTImage,
    max_offset: float = 30.0,
    seed: int | None = None,
) -> DCTImage:
    """Apply random brightness jitter.

    Samples a uniform offset in [-max_offset, +max_offset] and applies
    it to DC coefficients.

    Parameters
    ----------
    img : DCTImage
        Input image.
    max_offset : float
        Maximum brightness offset (pixel-value scale).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    DCTImage
    """
    if max_offset == 0:
        return DCTImage(
            y_coeffs=img.y_coeffs.copy(),
            cb_coeffs=img.cb_coeffs.copy() if img.cb_coeffs is not None else None,
            cr_coeffs=img.cr_coeffs.copy() if img.cr_coeffs is not None else None,
            quant_tables=img.quant_tables,
            width=img.width, height=img.height,
            comp_info=img.comp_info,
        )

    rng = np.random.RandomState(seed)
    offset = rng.uniform(-max_offset, max_offset)
    return adjust_brightness(img, offset=offset)


def contrast_jitter(
    img: DCTImage,
    max_factor: float = 0.3,
    seed: int | None = None,
) -> DCTImage:
    """Apply random contrast jitter.

    Samples a factor in [1 - max_factor, 1 + max_factor] and applies
    it to AC coefficients.

    Parameters
    ----------
    img : DCTImage
        Input image.
    max_factor : float
        Maximum deviation from factor=1.0.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    DCTImage
    """
    if max_factor == 0:
        return DCTImage(
            y_coeffs=img.y_coeffs.copy(),
            cb_coeffs=img.cb_coeffs.copy() if img.cb_coeffs is not None else None,
            cr_coeffs=img.cr_coeffs.copy() if img.cr_coeffs is not None else None,
            quant_tables=img.quant_tables,
            width=img.width, height=img.height,
            comp_info=img.comp_info,
        )

    rng = np.random.RandomState(seed)
    factor = 1.0 + rng.uniform(-max_factor, max_factor)
    factor = max(factor, 0.01)  # prevent zero/negative
    return adjust_contrast(img, factor=factor)
