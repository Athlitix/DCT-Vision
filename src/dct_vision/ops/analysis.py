"""Image analysis from DCT coefficients.

Blur detection, noise estimation, texture complexity, similarity,
histogram -- all computed directly from coefficients without pixel decode.
"""

from __future__ import annotations

import numpy as np

from dct_vision.core.dct_image import DCTImage
from dct_vision.utils.constants import BLOCK_SIZE


def detect_blur(img: DCTImage) -> float:
    """Detect how blurry an image is from DCT coefficients.

    Returns a score in [0, 1] where 1 = maximally blurry (no high-freq content)
    and 0 = very sharp (lots of high-freq content).

    The score is 1 - (high_freq_energy / total_energy).

    Parameters
    ----------
    img : DCTImage
        Input image.

    Returns
    -------
    float
        Blur score in [0, 1].
    """
    y = img.y_coeffs.astype(np.float64)

    # Total AC energy
    ac = y.copy()
    ac[:, :, 0, 0] = 0
    total_energy = np.sum(ac ** 2)

    if total_energy == 0:
        return 1.0

    # High-frequency energy (positions where u >= 4 or v >= 4)
    hf = np.zeros_like(y)
    hf[:, :, 4:, :] = y[:, :, 4:, :]
    hf[:, :, :, 4:] = y[:, :, :, 4:]
    hf[:, :, 0, 0] = 0
    hf_energy = np.sum(hf ** 2)

    return 1.0 - (hf_energy / total_energy)


def estimate_noise(img: DCTImage) -> float:
    """Estimate image noise level from high-frequency DCT coefficients.

    Uses the standard deviation of the highest-frequency coefficients
    (positions 6-7) as a noise proxy, since noise concentrates in
    high frequencies while signal concentrates in low frequencies.

    Parameters
    ----------
    img : DCTImage
        Input image.

    Returns
    -------
    float
        Estimated noise level (std of high-freq coefficients).
    """
    y = img.y_coeffs.astype(np.float64)
    # Highest frequency coefficients
    hf = y[:, :, 6:, 6:]
    return float(np.std(hf))


def texture_complexity(img: DCTImage) -> float:
    """Measure texture complexity / detail level.

    Returns the ratio of nonzero AC coefficients to total AC coefficients.
    Higher = more textured/detailed, lower = smoother/simpler.

    Parameters
    ----------
    img : DCTImage
        Input image.

    Returns
    -------
    float
        Complexity score in [0, 1].
    """
    y = img.y_coeffs.copy()
    y[:, :, 0, 0] = 0  # exclude DC
    total = y.size - img.y_coeffs.shape[0] * img.y_coeffs.shape[1]  # total AC
    nonzero = np.count_nonzero(y)
    if total == 0:
        return 0.0
    return float(nonzero / total)


def image_similarity(img1: DCTImage, img2: DCTImage) -> float:
    """Compare two images by their DCT coefficient correlation.

    Returns a similarity score in [0, 1] where 1 = identical
    coefficient distributions and 0 = completely uncorrelated.

    Uses normalized cross-correlation of the Y channel coefficients.

    Parameters
    ----------
    img1, img2 : DCTImage
        Images to compare (should have same dimensions).

    Returns
    -------
    float
        Similarity in [0, 1].
    """
    y1 = img1.y_coeffs.astype(np.float64).flatten()
    y2 = img2.y_coeffs.astype(np.float64).flatten()

    # Trim to common length
    n = min(len(y1), len(y2))
    y1 = y1[:n]
    y2 = y2[:n]

    # Normalized cross-correlation
    y1_norm = y1 - y1.mean()
    y2_norm = y2 - y2.mean()

    denom = np.sqrt(np.sum(y1_norm ** 2) * np.sum(y2_norm ** 2))
    if denom == 0:
        return 1.0 if np.allclose(y1, y2) else 0.0

    ncc = np.sum(y1_norm * y2_norm) / denom
    # Map from [-1, 1] to [0, 1]
    return float(np.clip((ncc + 1.0) / 2.0, 0.0, 1.0))


def histogram_from_dct(img: DCTImage, bins: int = 64) -> np.ndarray:
    """Approximate pixel histogram from DC coefficients.

    DC coefficient of each block is proportional to the block mean.
    This gives an approximate brightness histogram without pixel decode.

    Parameters
    ----------
    img : DCTImage
        Input image.
    bins : int
        Number of histogram bins.

    Returns
    -------
    np.ndarray
        Histogram counts, shape (bins,).
    """
    # DC coefficients represent block means (scaled)
    dc = img.y_coeffs[:, :, 0, 0].astype(np.float64).flatten()

    hist, _ = np.histogram(dc, bins=bins)
    return hist
