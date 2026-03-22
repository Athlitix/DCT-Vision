"""Perceptual image hashing (pHash) from DCT coefficients.

pHash is literally a DCT-based algorithm. Traditional implementations
decode to pixels, resize to 32x32, compute DCT, then threshold.
We already have the DCT coefficients -- we skip all the expensive steps.
"""

from __future__ import annotations

import numpy as np

from dct_vision.core.dct_image import DCTImage


def perceptual_hash(img: DCTImage, hash_size: int = 8) -> int:
    """Compute perceptual hash from DCT coefficients.

    Uses the low-frequency DCT coefficients from the Y (luminance) channel.
    Thresholds against the median to produce a binary hash.

    Parameters
    ----------
    img : DCTImage
        Input image.
    hash_size : int
        Hash dimension (produces hash_size^2 bit hash). Default 8 = 64-bit hash.

    Returns
    -------
    int
        Perceptual hash as integer.
    """
    y = img.y_coeffs.astype(np.float32)

    # Collect low-frequency coefficients from all blocks
    # Use DC and low-freq AC from a grid of blocks
    bh, bw = y.shape[:2]

    # Sample blocks evenly across the image
    row_indices = np.linspace(0, bh - 1, hash_size, dtype=int)
    col_indices = np.linspace(0, bw - 1, hash_size, dtype=int)

    # Extract low-frequency values (DC + first few AC)
    # Use DC coefficient as the primary feature
    features = np.zeros((hash_size, hash_size), dtype=np.float32)
    for i, ri in enumerate(row_indices):
        for j, ci in enumerate(col_indices):
            features[i, j] = y[ri, ci, 0, 0]  # DC coefficient

    # Threshold against median to get binary hash
    median = np.median(features)
    bits = (features > median).flatten()

    # Convert to integer
    hash_val = 0
    for bit in bits:
        hash_val = (hash_val << 1) | int(bit)

    return hash_val


def perceptual_hash_hex(img: DCTImage, hash_size: int = 8) -> str:
    """Compute perceptual hash as hex string.

    Parameters
    ----------
    img : DCTImage
        Input image.
    hash_size : int
        Hash dimension. Default 8 = 64-bit = 16 hex chars.

    Returns
    -------
    str
        Hex string representation of the hash.
    """
    h = perceptual_hash(img, hash_size)
    num_hex = (hash_size * hash_size + 3) // 4
    return f"{h:0{num_hex}x}"


def hamming_distance(hash1: int, hash2: int) -> int:
    """Compute Hamming distance between two hashes.

    Parameters
    ----------
    hash1, hash2 : int
        Perceptual hashes to compare.

    Returns
    -------
    int
        Number of differing bits.
    """
    return bin(hash1 ^ hash2).count("1")
