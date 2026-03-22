"""Photo editing operations in DCT domain.

Vignette, sepia, grayscale, posterize, solarize -- all directly
on DCT coefficients without pixel decode.
"""

from __future__ import annotations

import numpy as np

from dct_vision.core.dct_image import DCTImage
from dct_vision.utils.constants import BLOCK_SIZE


def vignette(img: DCTImage, strength: float = 1.0) -> DCTImage:
    """Apply vignette effect by darkening edge/corner blocks.

    Attenuates DC coefficients based on distance from image center.

    Parameters
    ----------
    img : DCTImage
        Input image.
    strength : float
        Vignette intensity. 0 = none, 1 = standard, 2 = heavy.

    Returns
    -------
    DCTImage
    """
    if strength == 0.0:
        return DCTImage(
            y_coeffs=img.y_coeffs.copy(),
            cb_coeffs=img.cb_coeffs.copy() if img.cb_coeffs is not None else None,
            cr_coeffs=img.cr_coeffs.copy() if img.cr_coeffs is not None else None,
            quant_tables=img.quant_tables,
            width=img.width, height=img.height,
            comp_info=img.comp_info,
            source_path=img._source_path,
        )

    bh, bw = img.y_coeffs.shape[:2]
    cy, cx = bh / 2.0, bw / 2.0
    max_dist = np.sqrt(cy**2 + cx**2)

    # Distance map for each block
    rows = np.arange(bh, dtype=np.float32) + 0.5
    cols = np.arange(bw, dtype=np.float32) + 0.5
    R, C = np.meshgrid(rows, cols, indexing="ij")
    dist = np.sqrt((R - cy)**2 + (C - cx)**2) / max_dist

    # Attenuation: 1.0 at center, decreasing toward edges
    atten = np.clip(1.0 - strength * dist**2, 0.0, 1.0).astype(np.float32)

    y = img.y_coeffs.astype(np.float32)
    # Apply attenuation to all coefficients (not just DC) for smooth vignette
    y = np.round(y * atten[:, :, np.newaxis, np.newaxis]).astype(np.int16)

    cb = img.cb_coeffs
    cr = img.cr_coeffs
    if cb is not None:
        # Apply to chroma too (proportional to block mapping)
        ch_bh, ch_bw = cb.shape[:2]
        ch_rows = np.linspace(0.5, bh - 0.5, ch_bh)
        ch_cols = np.linspace(0.5, bw - 0.5, ch_bw)
        CR, CC = np.meshgrid(ch_rows, ch_cols, indexing="ij")
        ch_dist = np.sqrt((CR - cy)**2 + (CC - cx)**2) / max_dist
        ch_atten = np.clip(1.0 - strength * ch_dist**2, 0.0, 1.0).astype(np.float32)
        cb = np.round(cb.astype(np.float32) * ch_atten[:, :, np.newaxis, np.newaxis]).astype(np.int16)
        cr = np.round(cr.astype(np.float32) * ch_atten[:, :, np.newaxis, np.newaxis]).astype(np.int16)

    return DCTImage(
        y_coeffs=y, cb_coeffs=cb, cr_coeffs=cr,
        quant_tables=img.quant_tables,
        width=img.width, height=img.height,
        comp_info=img.comp_info,
        source_path=img._source_path,
    )


def sepia(img: DCTImage) -> DCTImage:
    """Apply sepia tone by setting chroma channels to warm brown.

    Sets Cb/Cr DC to fixed warm values, scales AC to reduce chroma variation.

    Parameters
    ----------
    img : DCTImage
        Input color image.

    Returns
    -------
    DCTImage
    """
    if img.cb_coeffs is None:
        return DCTImage(
            y_coeffs=img.y_coeffs.copy(),
            cb_coeffs=None, cr_coeffs=None,
            quant_tables=img.quant_tables,
            width=img.width, height=img.height,
            comp_info=img.comp_info,
            source_path=img._source_path,
        )

    # Sepia in YCbCr: Cb slightly negative (less blue), Cr slightly positive (more red)
    cb = np.round(img.cb_coeffs.astype(np.float32) * 0.1).astype(np.int16)
    cr = np.round(img.cr_coeffs.astype(np.float32) * 0.1).astype(np.int16)

    # Set DC to sepia tint values
    cb[:, :, 0, 0] = -3   # slight blue reduction
    cr[:, :, 0, 0] = 4    # slight red boost

    return DCTImage(
        y_coeffs=img.y_coeffs.copy(),
        cb_coeffs=cb, cr_coeffs=cr,
        quant_tables=img.quant_tables,
        width=img.width, height=img.height,
        comp_info=img.comp_info,
        source_path=img._source_path,
    )


def grayscale(img: DCTImage) -> DCTImage:
    """Convert to grayscale by dropping chroma channels.

    Parameters
    ----------
    img : DCTImage
        Input image.

    Returns
    -------
    DCTImage
        Grayscale image (1 component).
    """
    return DCTImage(
        y_coeffs=img.y_coeffs.copy(),
        cb_coeffs=None, cr_coeffs=None,
        quant_tables=[img.quant_tables[0]],
        width=img.width, height=img.height,
        comp_info=[{"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": 0}],
        source_path=img._source_path,
    )


def posterize(img: DCTImage, levels: int = 4) -> DCTImage:
    """Posterize by aggressively requantizing coefficients.

    Reduces the number of distinct coefficient values, creating
    a poster-like effect with fewer tonal gradations.

    Parameters
    ----------
    img : DCTImage
        Input image.
    levels : int
        Number of quantization levels. Lower = more posterized. Must be >= 1.

    Returns
    -------
    DCTImage

    Raises
    ------
    ValueError
        If levels < 1.
    """
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")

    y = img.y_coeffs.astype(np.float32)
    # Quantize to fewer levels
    step = max(1.0, np.abs(y).max() / levels)
    y = np.round(y / step) * step
    y_out = np.round(y).astype(np.int16)

    cb = img.cb_coeffs
    cr = img.cr_coeffs
    if cb is not None:
        cb_f = cb.astype(np.float32)
        cb_step = max(1.0, np.abs(cb_f).max() / levels)
        cb = np.round(np.round(cb_f / cb_step) * cb_step).astype(np.int16)
        cr_f = cr.astype(np.float32)
        cr_step = max(1.0, np.abs(cr_f).max() / levels)
        cr = np.round(np.round(cr_f / cr_step) * cr_step).astype(np.int16)

    return DCTImage(
        y_coeffs=y_out, cb_coeffs=cb, cr_coeffs=cr,
        quant_tables=img.quant_tables,
        width=img.width, height=img.height,
        comp_info=img.comp_info,
        source_path=img._source_path,
    )


def solarize(img: DCTImage, threshold: int = 10) -> DCTImage:
    """Solarize by inverting coefficients above a threshold.

    Coefficients with absolute value above the threshold get negated,
    creating a surreal color-inversion effect.

    Parameters
    ----------
    img : DCTImage
        Input image.
    threshold : int
        Coefficient threshold for inversion.

    Returns
    -------
    DCTImage
    """
    y = img.y_coeffs.copy()
    mask = np.abs(y) > threshold
    y[mask] = -y[mask]

    cb = img.cb_coeffs
    cr = img.cr_coeffs
    if cb is not None:
        cb = cb.copy()
        cr = cr.copy()
        cb_mask = np.abs(cb) > threshold
        cb[cb_mask] = -cb[cb_mask]
        cr_mask = np.abs(cr) > threshold
        cr[cr_mask] = -cr[cr_mask]

    return DCTImage(
        y_coeffs=y, cb_coeffs=cb, cr_coeffs=cr,
        quant_tables=img.quant_tables,
        width=img.width, height=img.height,
        comp_info=img.comp_info,
        source_path=img._source_path,
    )
