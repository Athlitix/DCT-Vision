"""JPEG quality estimation from DCT coefficient statistics."""

from __future__ import annotations

import numpy as np

from dct_vision.core.dct_image import DCTImage
from dct_vision.utils.constants import LUMINANCE_QUANT_TABLE


def estimate_quality(img: DCTImage) -> int:
    """Estimate JPEG quality factor from quantization tables.

    Reverses the IJG quality-to-quantization-table formula to estimate
    the quality factor that produced the stored quantization table.

    Parameters
    ----------
    img : DCTImage
        Input image with quantization tables.

    Returns
    -------
    int
        Estimated quality factor (1-100).
    """
    qtable = img.quant_tables[0]

    # Compare against standard table to estimate the scale factor
    # scale = qtable / standard_table
    ratios = qtable / LUMINANCE_QUANT_TABLE
    avg_ratio = float(np.median(ratios))

    # Reverse the IJG formula:
    # if quality < 50: scale = 50 / quality  → quality = 50 / scale
    # if quality >= 50: scale = (100 - quality) / 50  → quality = 100 - 50 * scale
    if avg_ratio > 1.0:
        quality = int(round(50.0 / avg_ratio))
    else:
        quality = int(round(100.0 - 50.0 * avg_ratio))

    return max(1, min(100, quality))


def dct_stats(img: DCTImage) -> dict:
    """Compute statistics about DCT coefficients.

    Parameters
    ----------
    img : DCTImage
        Input image.

    Returns
    -------
    dict
        Keys: dc_mean, dc_std, ac_energy, num_nonzero_ac, total_ac.
    """
    y = img.y_coeffs.astype(np.float64)

    dc_values = y[:, :, 0, 0]
    ac_values = y.copy()
    ac_values[:, :, 0, 0] = 0

    total_ac = ac_values.size - dc_values.size

    return {
        "dc_mean": float(np.mean(dc_values)),
        "dc_std": float(np.std(dc_values)),
        "ac_energy": float(np.sum(ac_values**2)),
        "num_nonzero_ac": int(np.count_nonzero(ac_values)),
        "total_ac": total_ac,
    }
