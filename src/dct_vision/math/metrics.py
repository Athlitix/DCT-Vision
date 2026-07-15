"""Image quality metrics: MSE, PSNR, SSIM.

Pure numpy/scipy implementations (no OpenCV, no scikit-image) so they can
ship in the production library. SSIM follows Wang et al. (2004) with an
11x11 Gaussian window (sigma 1.5), matching scikit-image's ``gaussian_weights``
mode within tolerance.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

_MAX = 255.0
# SSIM stability constants (Wang et al. 2004), scaled to 8-bit range.
_C1 = (0.01 * _MAX) ** 2
_C2 = (0.03 * _MAX) ** 2


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean squared error between two images.

    Parameters
    ----------
    original, reconstructed : np.ndarray
        Images of identical shape. Any numeric dtype.

    Returns
    -------
    float
        Mean of squared per-element differences.
    """
    a = np.asarray(original, dtype=np.float64)
    b = np.asarray(reconstructed, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))


def psnr(original: np.ndarray, reconstructed: np.ndarray, max_value: float = _MAX) -> float:
    """Peak signal-to-noise ratio in decibels.

    Parameters
    ----------
    original, reconstructed : np.ndarray
        Images of identical shape.
    max_value : float
        Maximum possible pixel value (255 for 8-bit).

    Returns
    -------
    float
        PSNR in dB. ``inf`` when the images are identical.
    """
    error = mse(original, reconstructed)
    if error == 0.0:
        return float("inf")
    return float(10.0 * np.log10(max_value**2 / error))


def _ssim_channel(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM for a single 2D channel using a Gaussian window."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    win = dict(sigma=1.5, truncate=3.5, mode="reflect")

    mu_a = gaussian_filter(a, **win)
    mu_b = gaussian_filter(b, **win)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    var_a = gaussian_filter(a * a, **win) - mu_a_sq
    var_b = gaussian_filter(b * b, **win) - mu_b_sq
    cov_ab = gaussian_filter(a * b, **win) - mu_ab

    num = (2 * mu_ab + _C1) * (2 * cov_ab + _C2)
    den = (mu_a_sq + mu_b_sq + _C1) * (var_a + var_b + _C2)
    ssim_map = num / den

    # Match scikit-image: crop the border affected by the truncated window.
    pad = int(3.5 * 1.5 + 0.5)
    if ssim_map.shape[0] > 2 * pad and ssim_map.shape[1] > 2 * pad:
        ssim_map = ssim_map[pad:-pad, pad:-pad]
    return float(ssim_map.mean())


def ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Structural similarity index (Wang et al. 2004).

    Parameters
    ----------
    original, reconstructed : np.ndarray
        Images of identical shape, grayscale (H, W) or color (H, W, C).

    Returns
    -------
    float
        SSIM in [-1, 1]; 1.0 for identical images. For color images the
        mean over channels is returned.
    """
    a = np.asarray(original)
    b = np.asarray(reconstructed)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    if a.ndim == 2:
        return _ssim_channel(a, b)
    if a.ndim == 3:
        return float(np.mean([_ssim_channel(a[..., c], b[..., c]) for c in range(a.shape[2])]))
    raise ValueError(f"unsupported ndim: {a.ndim}")
