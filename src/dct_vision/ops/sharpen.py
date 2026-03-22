"""Sharpening in DCT domain via high-frequency coefficient boosting."""

from __future__ import annotations

import numpy as np

from dct_vision.core.dct_image import DCTImage
from dct_vision.utils.constants import BLOCK_SIZE


def sharpen(img: DCTImage, amount: float = 1.5) -> DCTImage:
    """Sharpen image by boosting high-frequency DCT coefficients.

    DC coefficient is preserved. AC coefficients are scaled by a weight
    that increases with frequency: weight(u,v) = 1 + (amount-1) * (u²+v²) / max_freq².

    Parameters
    ----------
    img : DCTImage
        Input image.
    amount : float
        Sharpening strength. 1.0 = no change, 2.0 = strong. Must be > 0.

    Returns
    -------
    DCTImage

    Raises
    ------
    ValueError
        If amount <= 0.
    """
    if amount <= 0:
        raise ValueError(f"amount must be > 0, got {amount}")

    if amount == 1.0:
        return DCTImage(
            y_coeffs=img.y_coeffs.copy(),
            cb_coeffs=img.cb_coeffs.copy() if img.cb_coeffs is not None else None,
            cr_coeffs=img.cr_coeffs.copy() if img.cr_coeffs is not None else None,
            quant_tables=img.quant_tables,
            width=img.width,
            height=img.height,
            comp_info=img.comp_info,
        )

    # Build sharpening weight matrix
    u = np.arange(BLOCK_SIZE, dtype=np.float32)
    v = np.arange(BLOCK_SIZE, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing="ij")
    max_freq_sq = float((BLOCK_SIZE - 1) ** 2 + (BLOCK_SIZE - 1) ** 2)
    freq_sq = U**2 + V**2

    # Linear ramp from 1.0 at DC to `amount` at highest frequency
    weights = (1.0 + (amount - 1.0) * freq_sq / max_freq_sq).astype(np.float32)
    weights[0, 0] = 1.0  # Preserve DC

    y_coeffs = np.round(img.y_coeffs.astype(np.float32) * weights).astype(np.int16)

    return DCTImage(
        y_coeffs=y_coeffs,
        cb_coeffs=img.cb_coeffs.copy() if img.cb_coeffs is not None else None,
        cr_coeffs=img.cr_coeffs.copy() if img.cr_coeffs is not None else None,
        quant_tables=img.quant_tables,
        width=img.width,
        height=img.height,
        comp_info=img.comp_info,
        source_path=img._source_path,
    )
