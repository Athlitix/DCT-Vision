"""Frequency-domain Gaussian noise injection.

Adds noise to high-frequency DCT coefficients while preserving DC (mean brightness).
"""

from __future__ import annotations

import numpy as np

from dct_vision.core.dct_image import DCTImage


def gaussian_noise(
    img: DCTImage,
    sigma: float = 3.0,
    seed: int | None = None,
) -> DCTImage:
    """Add Gaussian noise to high-frequency DCT coefficients.

    DC coefficient (block mean) is preserved. Noise is added to all
    AC coefficients with the given standard deviation.

    Parameters
    ----------
    img : DCTImage
        Input image.
    sigma : float
        Standard deviation of the Gaussian noise (in coefficient space).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    DCTImage
    """
    if sigma == 0:
        return DCTImage(
            y_coeffs=img.y_coeffs.copy(),
            cb_coeffs=img.cb_coeffs.copy() if img.cb_coeffs is not None else None,
            cr_coeffs=img.cr_coeffs.copy() if img.cr_coeffs is not None else None,
            quant_tables=img.quant_tables,
            width=img.width, height=img.height,
            comp_info=img.comp_info,
        )

    rng = np.random.RandomState(seed)

    y = img.y_coeffs.astype(np.float32)
    noise = rng.normal(0, sigma, y.shape).astype(np.float32)
    noise[:, :, 0, 0] = 0  # Preserve DC
    y = np.round(y + noise).astype(np.int16)

    return DCTImage(
        y_coeffs=y,
        cb_coeffs=img.cb_coeffs.copy() if img.cb_coeffs is not None else None,
        cr_coeffs=img.cr_coeffs.copy() if img.cr_coeffs is not None else None,
        quant_tables=img.quant_tables,
        width=img.width, height=img.height,
        comp_info=img.comp_info,
    )
