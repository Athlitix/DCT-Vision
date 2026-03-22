"""DCT/IDCT utilities for 8x8 block processing."""

import numpy as np
from scipy.fft import dctn, idctn

from dct_vision.utils.constants import BLOCK_SIZE


def dct2(block: np.ndarray) -> np.ndarray:
    """Compute 2D DCT of an 8x8 block.

    Parameters
    ----------
    block : np.ndarray
        Input block, shape (8, 8), dtype float32.

    Returns
    -------
    np.ndarray
        DCT coefficients, shape (8, 8), dtype float32.
    """
    return dctn(block, type=2, norm="ortho").astype(np.float32)


def idct2(coeffs: np.ndarray) -> np.ndarray:
    """Compute 2D inverse DCT of an 8x8 coefficient block.

    Parameters
    ----------
    coeffs : np.ndarray
        DCT coefficients, shape (8, 8), dtype float32.

    Returns
    -------
    np.ndarray
        Reconstructed block, shape (8, 8), dtype float32.
    """
    return idctn(coeffs, type=2, norm="ortho").astype(np.float32)


def blockwise_dct(channel: np.ndarray) -> np.ndarray:
    """Apply 2D DCT to all 8x8 blocks in a channel.

    Parameters
    ----------
    channel : np.ndarray
        Single channel image, shape (H, W) where H and W are multiples of 8.

    Returns
    -------
    np.ndarray
        DCT coefficients, shape (H//8, W//8, 8, 8), dtype float32.

    Raises
    ------
    ValueError
        If dimensions are not multiples of 8.
    """
    h, w = channel.shape
    if h % BLOCK_SIZE != 0 or w % BLOCK_SIZE != 0:
        raise ValueError(
            f"Channel dimensions ({h}, {w}) must be multiples of {BLOCK_SIZE}."
        )

    bh = h // BLOCK_SIZE
    bw = w // BLOCK_SIZE
    coeffs = np.zeros((bh, bw, BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)

    for i in range(bh):
        for j in range(bw):
            block = channel[
                i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE,
                j * BLOCK_SIZE : (j + 1) * BLOCK_SIZE,
            ]
            coeffs[i, j] = dct2(block)

    return coeffs


def blockwise_idct(coeffs: np.ndarray) -> np.ndarray:
    """Apply 2D inverse DCT to reconstruct a channel from block coefficients.

    Parameters
    ----------
    coeffs : np.ndarray
        DCT coefficients, shape (bh, bw, 8, 8).

    Returns
    -------
    np.ndarray
        Reconstructed channel, shape (bh*8, bw*8), dtype float32.
    """
    bh, bw = coeffs.shape[:2]
    h = bh * BLOCK_SIZE
    w = bw * BLOCK_SIZE
    channel = np.zeros((h, w), dtype=np.float32)

    for i in range(bh):
        for j in range(bw):
            channel[
                i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE,
                j * BLOCK_SIZE : (j + 1) * BLOCK_SIZE,
            ] = idct2(coeffs[i, j])

    return channel
