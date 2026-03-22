"""Block-aligned crop in DCT domain.

Crops by selecting a subset of DCT blocks -- zero pixel decoding needed.
"""

from __future__ import annotations

import numpy as np

from dct_vision.core.dct_image import DCTImage
from dct_vision.utils.constants import BLOCK_SIZE


def block_crop(
    img: DCTImage,
    block_row: int,
    block_col: int,
    block_rows: int,
    block_cols: int,
) -> DCTImage:
    """Crop image to a block-aligned region.

    Parameters
    ----------
    img : DCTImage
        Input image.
    block_row, block_col : int
        Top-left block coordinates.
    block_rows, block_cols : int
        Number of blocks to include in each dimension.

    Returns
    -------
    DCTImage
        Cropped image.

    Raises
    ------
    ValueError
        If crop region extends beyond image boundaries.
    """
    bh, bw = img.y_coeffs.shape[:2]

    if block_row + block_rows > bh or block_col + block_cols > bw:
        raise ValueError(
            f"Crop region ({block_row}:{block_row+block_rows}, "
            f"{block_col}:{block_col+block_cols}) exceeds image "
            f"block dimensions ({bh}, {bw})."
        )

    y = img.y_coeffs[
        block_row : block_row + block_rows,
        block_col : block_col + block_cols,
    ].copy()

    cb = None
    cr = None
    if img.cb_coeffs is not None:
        # Handle chroma subsampling -- chroma blocks may be at different scale
        ch_bh, ch_bw = img.cb_coeffs.shape[:2]
        ch_scale_h = ch_bh / bh
        ch_scale_w = ch_bw / bw

        ch_r = int(block_row * ch_scale_h)
        ch_c = int(block_col * ch_scale_w)
        ch_rows = max(1, int(block_rows * ch_scale_h))
        ch_cols = max(1, int(block_cols * ch_scale_w))

        # Clamp
        ch_rows = min(ch_rows, ch_bh - ch_r)
        ch_cols = min(ch_cols, ch_bw - ch_c)

        cb = img.cb_coeffs[ch_r : ch_r + ch_rows, ch_c : ch_c + ch_cols].copy()
        cr = img.cr_coeffs[ch_r : ch_r + ch_rows, ch_c : ch_c + ch_cols].copy()

    return DCTImage(
        y_coeffs=y, cb_coeffs=cb, cr_coeffs=cr,
        quant_tables=img.quant_tables,
        width=block_cols * BLOCK_SIZE,
        height=block_rows * BLOCK_SIZE,
        comp_info=img.comp_info,
    )
