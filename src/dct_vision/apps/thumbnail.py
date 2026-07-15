"""Instant thumbnail generation from DCT DC coefficients.

The DC coefficient of each 8x8 block equals the block mean times 8 (orthonormal
DCT). So the grid of DC coefficients is *already* an 8x-downscaled version of
the image -- a thumbnail obtainable with one array operation and no IDCT.
"""

from __future__ import annotations

import numpy as np

from dct_vision.core.dct_image import DCTImage
from dct_vision.utils.constants import BLOCK_SIZE


def _dc_channel(coeffs: np.ndarray, qtable: np.ndarray) -> np.ndarray:
    """Pixel-domain block means from DC coefficients: mean = DC/8 + 128."""
    dc = coeffs[:, :, 0, 0].astype(np.float32) * qtable[0, 0]
    return dc / BLOCK_SIZE + 128.0


def dc_thumbnail(img: DCTImage, size: int | None = None) -> np.ndarray:
    """Generate a thumbnail from DC coefficients (no IDCT).

    Produces an (bh, bw) grayscale or (bh, bw, 3) RGB array -- the 1/8-scale DC
    image -- optionally resized so its longer side is ``size`` pixels.

    Parameters
    ----------
    img : DCTImage
        Input image.
    size : int, optional
        If given, resize so the longer side equals ``size`` (bilinear via
        Pillow), preserving aspect ratio.

    Returns
    -------
    np.ndarray
        uint8 thumbnail, shape (H, W) or (H, W, 3).
    """
    qtables = img.quant_tables

    def qt(idx_default):
        idx = idx_default
        if img.comp_info:
            idx = img.comp_info[min(idx_default, len(img.comp_info) - 1)].get("quant_tbl_no", idx_default)
        return qtables[min(idx, len(qtables) - 1)]

    y = _dc_channel(img.y_coeffs, qt(0))

    if img.cb_coeffs is None:
        thumb = np.clip(y, 0, 255).astype(np.uint8)
    else:
        cb = _dc_channel(img.cb_coeffs, qt(1))
        cr = _dc_channel(img.cr_coeffs, qt(2 if img.comp_info and len(img.comp_info) > 2 else 1))
        # Upsample chroma DC grids to the luma grid if subsampled.
        bh, bw = y.shape
        if cb.shape != y.shape:
            cb = np.repeat(np.repeat(cb, max(1, bh // cb.shape[0]), 0), max(1, bw // cb.shape[1]), 1)[:bh, :bw]
            cr = np.repeat(np.repeat(cr, max(1, bh // cr.shape[0]), 0), max(1, bw // cr.shape[1]), 1)[:bh, :bw]
        from dct_vision.math.colorspace import ycbcr_to_rgb

        ycbcr = np.stack([y, cb, cr], axis=-1).astype(np.float32)
        thumb = np.clip(ycbcr_to_rgb(ycbcr), 0, 255).astype(np.uint8)

    if size is not None:
        from PIL import Image

        h, w = thumb.shape[:2]
        scale = size / max(h, w)
        new = (max(1, round(w * scale)), max(1, round(h * scale)))
        mode = "L" if thumb.ndim == 2 else "RGB"
        thumb = np.array(Image.fromarray(thumb, mode=mode).resize(new, Image.BILINEAR))

    return thumb
