"""Lossless geometric transforms in the DCT domain.

Rotation by multiples of 90 degrees and transpose are exact coefficient
permutations -- no IDCT/DCT, no pixel decode, and no quality loss (this is
what ``jpegtran`` does for lossless JPEG rotation).

Math
----
For a separable orthonormal 8x8 DCT with coefficients ``C[u, v]``:

- Transpose (main diagonal): transposing the spatial block transposes the
  coefficients, ``C'[u, v] = C[v, u]``, and the block grid transposes too.
- Horizontal flip: reverse block columns, negate odd horizontal frequencies
  (multiply by ``(-1)**v``).
- Vertical flip: reverse block rows, negate odd vertical frequencies
  (multiply by ``(-1)**u``).

Rotations are compositions of transpose and flips:

- 90 clockwise  = horizontal_flip(transpose)
- 90 counter-cw = vertical_flip(transpose)   (== rotate 270 clockwise)
- 180           = reverse both block axes, multiply by ``(-1)**(u+v)``
"""

from __future__ import annotations

import numpy as np

from dct_vision.augment.flip import horizontal_flip, vertical_flip
from dct_vision.core.dct_image import DCTImage

# Sign mask (-1)**(u+v) over an 8x8 block, used for the 180 degree rotation.
_uu, _vv = np.mgrid[0:8, 0:8]
_SIGN_180 = np.where((_uu + _vv) % 2 == 0, 1.0, -1.0).astype(np.float32)


def _transpose_comp_info(comp_info: list[dict] | None) -> list[dict] | None:
    """Swap horizontal/vertical sampling factors (and block counts) so chroma
    stays aligned after a transpose."""
    if comp_info is None:
        return None
    swapped = []
    for c in comp_info:
        d = dict(c)
        d["h_samp_factor"], d["v_samp_factor"] = c.get("v_samp_factor", 1), c.get("h_samp_factor", 1)
        if "width_in_blocks" in c or "height_in_blocks" in c:
            d["width_in_blocks"] = c.get("height_in_blocks")
            d["height_in_blocks"] = c.get("width_in_blocks")
        swapped.append(d)
    return swapped


def _transpose_coeffs(coeffs: np.ndarray | None) -> np.ndarray | None:
    """Transpose block grid and each block: C'[j,i,v,u] = C[i,j,u,v]."""
    if coeffs is None:
        return None
    return np.ascontiguousarray(coeffs.transpose(1, 0, 3, 2))


def transpose(img: DCTImage) -> DCTImage:
    """Transpose an image about its main diagonal, losslessly.

    Because transposing a block swaps coefficient positions (u, v) -> (v, u) and
    the JPEG quantization tables are not symmetric, the quant tables must be
    transposed too so each coefficient is dequantized by the right step. The
    native lossless writer templates off the source JPEG's (un-transposed)
    tables, so ``source_path`` is dropped to force the correct table on save.

    Parameters
    ----------
    img : DCTImage
        Input image.

    Returns
    -------
    DCTImage
        Transposed image (width and height swapped).
    """
    return DCTImage(
        y_coeffs=_transpose_coeffs(img.y_coeffs),
        cb_coeffs=_transpose_coeffs(img.cb_coeffs),
        cr_coeffs=_transpose_coeffs(img.cr_coeffs),
        quant_tables=[np.ascontiguousarray(q.T) for q in img.quant_tables],
        width=img.height,
        height=img.width,
        comp_info=_transpose_comp_info(img.comp_info),
        source_path=None,
    )


def rotate180(img: DCTImage) -> DCTImage:
    """Rotate an image 180 degrees, losslessly.

    Parameters
    ----------
    img : DCTImage
        Input image.

    Returns
    -------
    DCTImage
        Rotated image (same dimensions).
    """
    def _rot(coeffs):
        if coeffs is None:
            return None
        return np.ascontiguousarray(coeffs[::-1, ::-1, :, :] * _SIGN_180)

    return DCTImage(
        y_coeffs=_rot(img.y_coeffs),
        cb_coeffs=_rot(img.cb_coeffs),
        cr_coeffs=_rot(img.cr_coeffs),
        quant_tables=img.quant_tables,
        width=img.width,
        height=img.height,
        comp_info=img.comp_info,
        source_path=img._source_path,
    )


def rotate90(img: DCTImage) -> DCTImage:
    """Rotate an image 90 degrees clockwise, losslessly.

    Parameters
    ----------
    img : DCTImage
        Input image.

    Returns
    -------
    DCTImage
        Rotated image (width and height swapped).
    """
    return horizontal_flip(transpose(img))


def rotate270(img: DCTImage) -> DCTImage:
    """Rotate an image 270 degrees clockwise (90 counter-clockwise), losslessly.

    Parameters
    ----------
    img : DCTImage
        Input image.

    Returns
    -------
    DCTImage
        Rotated image (width and height swapped).
    """
    return vertical_flip(transpose(img))


def rotate(img: DCTImage, degrees: int) -> DCTImage:
    """Rotate an image clockwise by a multiple of 90 degrees, losslessly.

    Parameters
    ----------
    img : DCTImage
        Input image.
    degrees : int
        Clockwise rotation in degrees; one of 0, 90, 180, 270 (or -90, -180,
        -270 for counter-clockwise).

    Returns
    -------
    DCTImage
        Rotated image.

    Raises
    ------
    ValueError
        If ``degrees`` is not a multiple of 90.
    """
    d = degrees % 360
    if d == 0:
        return DCTImage(
            y_coeffs=img.y_coeffs.copy(),
            cb_coeffs=img.cb_coeffs.copy() if img.cb_coeffs is not None else None,
            cr_coeffs=img.cr_coeffs.copy() if img.cr_coeffs is not None else None,
            quant_tables=img.quant_tables,
            width=img.width,
            height=img.height,
            comp_info=img.comp_info,
            source_path=img._source_path,
        )
    if d == 90:
        return rotate90(img)
    if d == 180:
        return rotate180(img)
    if d == 270:
        return rotate270(img)
    raise ValueError(f"degrees must be a multiple of 90, got {degrees}")
