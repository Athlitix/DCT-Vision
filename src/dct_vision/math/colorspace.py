"""RGB <-> YCbCr colorspace conversion (ITU-R BT.601)."""

import numpy as np


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to YCbCr.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image, shape (H, W, 3), values in [0, 255], dtype float32.

    Returns
    -------
    np.ndarray
        YCbCr image, shape (H, W, 3), dtype float32.
        Y in [0, 255], Cb/Cr in [0, 255] (centered at 128).
    """
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0

    return np.stack([y, cb, cr], axis=-1).astype(np.float32)


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """Convert YCbCr image to RGB.

    Parameters
    ----------
    ycbcr : np.ndarray
        YCbCr image, shape (H, W, 3), dtype float32.

    Returns
    -------
    np.ndarray
        RGB image, shape (H, W, 3), clamped to [0, 255], dtype float32.
    """
    y = ycbcr[..., 0]
    cb = ycbcr[..., 1] - 128.0
    cr = ycbcr[..., 2] - 128.0

    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb

    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    return np.clip(rgb, 0.0, 255.0)
