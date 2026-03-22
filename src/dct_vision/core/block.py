"""8x8 block utilities for JPEG DCT processing."""

from typing import Iterator
import numpy as np

from dct_vision.utils.constants import BLOCK_SIZE
import math


def pixel_to_block_coords(row: int, col: int) -> tuple[int, int]:
    """Convert pixel coordinates to block coordinates."""
    return row // BLOCK_SIZE, col // BLOCK_SIZE


def block_to_pixel_coords(block_row: int, block_col: int) -> tuple[int, int]:
    """Convert block coordinates to top-left pixel coordinates."""
    return block_row * BLOCK_SIZE, block_col * BLOCK_SIZE


def pad_to_block_multiple(channel: np.ndarray) -> np.ndarray:
    """Pad a channel to dimensions that are multiples of BLOCK_SIZE.

    Parameters
    ----------
    channel : np.ndarray
        Single channel, shape (H, W).

    Returns
    -------
    np.ndarray
        Zero-padded channel with dimensions rounded up to multiples of 8.
    """
    h, w = channel.shape
    new_h = math.ceil(h / BLOCK_SIZE) * BLOCK_SIZE
    new_w = math.ceil(w / BLOCK_SIZE) * BLOCK_SIZE

    if new_h == h and new_w == w:
        return channel

    padded = np.zeros((new_h, new_w), dtype=channel.dtype)
    padded[:h, :w] = channel
    return padded


def iter_blocks(
    channel: np.ndarray,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Iterate over 8x8 blocks of a channel.

    Parameters
    ----------
    channel : np.ndarray
        Single channel, shape (H, W). Must be divisible by BLOCK_SIZE.

    Yields
    ------
    tuple[int, int, np.ndarray]
        (block_row, block_col, 8x8 block).

    Raises
    ------
    ValueError
        If channel dimensions are not multiples of BLOCK_SIZE.
    """
    h, w = channel.shape
    if h % BLOCK_SIZE != 0 or w % BLOCK_SIZE != 0:
        raise ValueError(
            f"Channel dimensions ({h}, {w}) must be multiples of {BLOCK_SIZE}. "
            f"Use pad_to_block_multiple() first."
        )

    for i in range(h // BLOCK_SIZE):
        for j in range(w // BLOCK_SIZE):
            block = channel[
                i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE,
                j * BLOCK_SIZE : (j + 1) * BLOCK_SIZE,
            ]
            yield i, j, block
