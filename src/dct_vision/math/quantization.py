"""JPEG quantization and dequantization utilities."""

import numpy as np


def quantize(coeffs: np.ndarray, qtable: np.ndarray) -> np.ndarray:
    """Quantize DCT coefficients by dividing and rounding.

    Parameters
    ----------
    coeffs : np.ndarray
        DCT coefficients, shape (..., 8, 8).
    qtable : np.ndarray
        Quantization table, shape (8, 8).

    Returns
    -------
    np.ndarray
        Quantized coefficients, dtype float32.
    """
    return np.round(coeffs / qtable).astype(np.float32)


def dequantize(coeffs: np.ndarray, qtable: np.ndarray) -> np.ndarray:
    """Dequantize coefficients by multiplying with quantization table.

    Parameters
    ----------
    coeffs : np.ndarray
        Quantized DCT coefficients, shape (..., 8, 8).
    qtable : np.ndarray
        Quantization table, shape (8, 8).

    Returns
    -------
    np.ndarray
        Dequantized coefficients, dtype float32.
    """
    return (coeffs * qtable).astype(np.float32)


def quality_to_scale_factor(quality: int) -> float:
    """Convert JPEG quality (1-100) to quantization scale factor.

    Follows the IJG (Independent JPEG Group) formula used by libjpeg.

    Parameters
    ----------
    quality : int
        JPEG quality factor, 1 (worst) to 100 (best).

    Returns
    -------
    float
        Scale factor to multiply base quantization table by.

    Raises
    ------
    ValueError
        If quality is not in [1, 100].
    """
    if quality < 1 or quality > 100:
        raise ValueError(f"Quality must be in [1, 100], got {quality}.")

    if quality < 50:
        return 50.0 / quality
    else:
        return (100.0 - quality) / 50.0


def scale_quant_table(qtable: np.ndarray, quality: int) -> np.ndarray:
    """Scale a quantization table for a given JPEG quality factor.

    Parameters
    ----------
    qtable : np.ndarray
        Base quantization table, shape (8, 8).
    quality : int
        JPEG quality factor, 1-100.

    Returns
    -------
    np.ndarray
        Scaled quantization table with values clamped to >= 1, dtype float32.
    """
    sf = quality_to_scale_factor(quality)
    scaled = np.floor(qtable * sf + 0.5)
    return np.clip(scaled, 1.0, 255.0).astype(np.float32)
