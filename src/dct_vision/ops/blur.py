"""Gaussian blur in DCT frequency domain.

Applies blur by multiplying DCT coefficients with a Gaussian envelope.
This is equivalent to spatial Gaussian convolution via the convolution theorem:
    f * g in spatial domain = F . G in frequency domain
"""

from __future__ import annotations
from functools import lru_cache

import numpy as np

from dct_vision.core.dct_image import DCTImage
from dct_vision.utils.constants import BLOCK_SIZE


@lru_cache(maxsize=32)
def gaussian_envelope(sigma: float, block_size: int = BLOCK_SIZE) -> np.ndarray:
    """Generate frequency-domain Gaussian filter for DCT coefficients.

    Parameters
    ----------
    sigma : float
        Standard deviation controlling blur strength. Must be > 0.
    block_size : int
        Size of DCT block (default 8).

    Returns
    -------
    np.ndarray
        Weight matrix, shape (block_size, block_size), dtype float32.
        Values range from 1.0 (DC) to near-zero (high frequencies).

    Raises
    ------
    ValueError
        If sigma <= 0.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    u = np.arange(block_size, dtype=np.float32)
    v = np.arange(block_size, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing="ij")
    envelope = np.exp(-(U**2 + V**2) / (2 * sigma**2)).astype(np.float32)

    # Normalize so DC component weight is 1.0
    envelope /= envelope[0, 0]

    return envelope


def _apply_envelope(coeffs: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    """Apply frequency envelope to block coefficients."""
    # coeffs shape: (bh, bw, 8, 8), envelope shape: (8, 8)
    result = coeffs.astype(np.float32) * envelope
    return np.round(result).astype(np.int16)


def blur(
    img: DCTImage,
    sigma: float,
    channels: str = "all",
    cross_block: bool = False,
) -> DCTImage:
    """Apply Gaussian blur in DCT domain.

    Parameters
    ----------
    img : DCTImage
        Input image.
    sigma : float
        Gaussian blur sigma. Larger = more blur. Must be > 0.
    channels : str
        'all' (default), 'luma', or 'chroma'.
    cross_block : bool
        Use 3x3 block neighborhood strategy for cross-boundary smoothness.
        Recommended for sigma > 2.0 to avoid block seam artifacts.
        Slower but higher quality.

    Returns
    -------
    DCTImage
        Blurred image.

    Raises
    ------
    ValueError
        If sigma <= 0.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    if cross_block:
        from dct_vision.ops.cross_block import cross_block_blur
        return cross_block_blur(img, sigma=sigma, channels=channels)

    envelope = gaussian_envelope(sigma)

    # Blur luma
    if channels in ("all", "luma"):
        y_coeffs = _apply_envelope(img.y_coeffs, envelope)
    else:
        y_coeffs = img.y_coeffs.copy()

    # Blur chroma
    if img.cb_coeffs is not None and channels in ("all", "chroma"):
        cb_coeffs = _apply_envelope(img.cb_coeffs, envelope)
        cr_coeffs = _apply_envelope(img.cr_coeffs, envelope)
    else:
        cb_coeffs = img.cb_coeffs.copy() if img.cb_coeffs is not None else None
        cr_coeffs = img.cr_coeffs.copy() if img.cr_coeffs is not None else None

    return DCTImage(
        y_coeffs=y_coeffs,
        cb_coeffs=cb_coeffs,
        cr_coeffs=cr_coeffs,
        quant_tables=img.quant_tables,
        width=img.width,
        height=img.height,
        comp_info=img.comp_info,
        source_path=img._source_path,
    )
