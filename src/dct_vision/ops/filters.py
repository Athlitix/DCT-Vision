"""Classical CV filters in DCT domain.

Sobel, Scharr, box blur, emboss, band-pass, and unsharp mask --
all operating directly on DCT coefficients.
"""

from __future__ import annotations
from functools import lru_cache

import numpy as np

from dct_vision.core.dct_image import DCTImage
from dct_vision.ops.blur import gaussian_envelope
from dct_vision.utils.constants import BLOCK_SIZE


# -- Weight matrix builders --

@lru_cache(maxsize=8)
def _sobel_weights(direction: str) -> np.ndarray:
    """Sobel-like frequency weights for gradient approximation.

    Sobel computes first derivative. In frequency domain,
    differentiation maps to multiplication by frequency index.
    """
    u = np.arange(BLOCK_SIZE, dtype=np.float32)
    v = np.arange(BLOCK_SIZE, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing="ij")
    max_freq = float(BLOCK_SIZE - 1)

    if direction == "horizontal":
        # d/dx -> multiply by horizontal frequency u
        weights = U / max_freq
    elif direction == "vertical":
        # d/dy -> multiply by vertical frequency v
        weights = V / max_freq
    else:
        raise ValueError(f"direction must be 'horizontal', 'vertical', or 'both', got '{direction}'")

    weights[0, 0] = 0  # suppress DC
    return weights.astype(np.float32)


@lru_cache(maxsize=8)
def _scharr_weights(direction: str) -> np.ndarray:
    """Scharr-like weights -- emphasizes accuracy over Sobel.

    Scharr uses a 3x3 kernel [3,10,3] vs Sobel's [1,2,1].
    In frequency domain, this maps to a slightly different
    frequency emphasis curve.
    """
    u = np.arange(BLOCK_SIZE, dtype=np.float32)
    v = np.arange(BLOCK_SIZE, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing="ij")
    max_freq = float(BLOCK_SIZE - 1)

    # Scharr emphasizes mid-frequencies more than Sobel
    # Weight = freq * (1 + 0.5 * cos(pi * cross_freq / max))
    if direction == "horizontal":
        cross_emphasis = 1.0 + 0.5 * np.cos(np.pi * V / max_freq)
        weights = (U / max_freq) * cross_emphasis
    elif direction == "vertical":
        cross_emphasis = 1.0 + 0.5 * np.cos(np.pi * U / max_freq)
        weights = (V / max_freq) * cross_emphasis
    else:
        raise ValueError(f"direction must be 'horizontal', 'vertical', or 'both', got '{direction}'")

    weights[0, 0] = 0
    return weights.astype(np.float32)


@lru_cache(maxsize=8)
def _box_blur_envelope(radius: int) -> np.ndarray:
    """Frequency envelope for box blur (sinc-like attenuation).

    A box kernel in spatial domain has a sinc transform in frequency domain.
    We approximate this as stronger attenuation of high frequencies
    proportional to the radius.
    """
    u = np.arange(BLOCK_SIZE, dtype=np.float32)
    v = np.arange(BLOCK_SIZE, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing="ij")

    # sinc(u * radius / N) * sinc(v * radius / N)
    # For u=0 or v=0, sinc(0) = 1
    scale = radius / BLOCK_SIZE
    with np.errstate(divide="ignore", invalid="ignore"):
        su = np.where(U == 0, 1.0, np.sinc(U * scale))
        sv = np.where(V == 0, 1.0, np.sinc(V * scale))

    envelope = (su * sv).astype(np.float32)
    envelope[0, 0] = 1.0  # preserve DC
    return envelope


@lru_cache(maxsize=8)
def _emboss_weights(angle: float) -> np.ndarray:
    """Emboss weights -- directional high-pass.

    Emboss emphasizes edges in a specific direction by weighting
    frequency components along that angle.
    """
    u = np.arange(BLOCK_SIZE, dtype=np.float32)
    v = np.arange(BLOCK_SIZE, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing="ij")
    max_freq = float(BLOCK_SIZE - 1)

    rad = np.radians(angle)
    # Project frequency onto the angle direction
    projection = (U * np.cos(rad) + V * np.sin(rad)) / max_freq
    weights = np.abs(projection).astype(np.float32)
    weights[0, 0] = 0  # suppress DC
    return weights


def _apply_weights_grayscale(img: DCTImage, weights: np.ndarray) -> DCTImage:
    """Apply weight matrix to luma, return grayscale DCTImage."""
    y = np.round(img.y_coeffs.astype(np.float32) * weights).astype(np.int16)
    qtable = img.quant_tables[0]
    return DCTImage(
        y_coeffs=y, cb_coeffs=None, cr_coeffs=None,
        quant_tables=[qtable],
        width=img.width, height=img.height,
        comp_info=[{"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": 0}],
        source_path=img._source_path,
    )


# -- Public API --

def sobel(img: DCTImage, direction: str = "both") -> DCTImage:
    """Sobel edge detection in DCT domain.

    Parameters
    ----------
    img : DCTImage
        Input image.
    direction : str
        'horizontal', 'vertical', or 'both' (magnitude).

    Returns
    -------
    DCTImage
        Grayscale edge map.
    """
    if direction == "both":
        h_w = _sobel_weights("horizontal")
        v_w = _sobel_weights("vertical")
        y = img.y_coeffs.astype(np.float32)
        h_result = y * h_w
        v_result = y * v_w
        magnitude = np.sqrt(h_result**2 + v_result**2)
        y_out = np.round(magnitude).astype(np.int16)
        qtable = img.quant_tables[0]
        return DCTImage(
            y_coeffs=y_out, cb_coeffs=None, cr_coeffs=None,
            quant_tables=[qtable],
            width=img.width, height=img.height,
            comp_info=[{"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": 0}],
            source_path=img._source_path,
        )

    weights = _sobel_weights(direction)
    return _apply_weights_grayscale(img, weights)


def scharr(img: DCTImage, direction: str = "both") -> DCTImage:
    """Scharr edge detection in DCT domain.

    More accurate gradient estimation than Sobel, especially
    for diagonal edges.

    Parameters
    ----------
    img : DCTImage
        Input image.
    direction : str
        'horizontal', 'vertical', or 'both' (magnitude).

    Returns
    -------
    DCTImage
        Grayscale edge map.
    """
    if direction == "both":
        h_w = _scharr_weights("horizontal")
        v_w = _scharr_weights("vertical")
        y = img.y_coeffs.astype(np.float32)
        magnitude = np.sqrt((y * h_w)**2 + (y * v_w)**2)
        y_out = np.round(magnitude).astype(np.int16)
        qtable = img.quant_tables[0]
        return DCTImage(
            y_coeffs=y_out, cb_coeffs=None, cr_coeffs=None,
            quant_tables=[qtable],
            width=img.width, height=img.height,
            comp_info=[{"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": 0}],
            source_path=img._source_path,
        )

    weights = _scharr_weights(direction)
    return _apply_weights_grayscale(img, weights)


def box_blur(img: DCTImage, radius: int = 2) -> DCTImage:
    """Box blur (averaging filter) in DCT domain.

    Parameters
    ----------
    img : DCTImage
        Input image.
    radius : int
        Blur radius in pixels. Must be >= 1.

    Returns
    -------
    DCTImage

    Raises
    ------
    ValueError
        If radius < 1.
    """
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")

    envelope = _box_blur_envelope(radius)

    y = np.round(img.y_coeffs.astype(np.float32) * envelope).astype(np.int16)

    cb = None
    cr = None
    if img.cb_coeffs is not None:
        cb = np.round(img.cb_coeffs.astype(np.float32) * envelope).astype(np.int16)
        cr = np.round(img.cr_coeffs.astype(np.float32) * envelope).astype(np.int16)

    return DCTImage(
        y_coeffs=y, cb_coeffs=cb, cr_coeffs=cr,
        quant_tables=img.quant_tables,
        width=img.width, height=img.height,
        comp_info=img.comp_info,
        source_path=img._source_path,
    )


def emboss(img: DCTImage, angle: float = 45.0) -> DCTImage:
    """Emboss filter in DCT domain.

    Creates a relief/3D effect by emphasizing edges in a specific direction.

    Parameters
    ----------
    img : DCTImage
        Input image.
    angle : float
        Emboss direction in degrees (0 = horizontal, 90 = vertical).

    Returns
    -------
    DCTImage
        Grayscale embossed image.
    """
    weights = _emboss_weights(angle)
    return _apply_weights_grayscale(img, weights)


def bandpass(img: DCTImage, low_cutoff: int = 1, high_cutoff: int = 6) -> DCTImage:
    """Band-pass filter in DCT domain.

    Keeps only frequency components between low_cutoff and high_cutoff,
    zeroing everything outside that range. This is a natural operation
    in frequency domain with no spatial equivalent in OpenCV.

    Parameters
    ----------
    img : DCTImage
        Input image.
    low_cutoff : int
        Lowest frequency index to keep (0 = DC).
    high_cutoff : int
        Highest frequency index to keep (7 = max for 8x8 blocks).

    Returns
    -------
    DCTImage

    Raises
    ------
    ValueError
        If low_cutoff > high_cutoff.
    """
    if low_cutoff > high_cutoff:
        raise ValueError(
            f"low_cutoff ({low_cutoff}) must be <= high_cutoff ({high_cutoff})"
        )

    u = np.arange(BLOCK_SIZE, dtype=np.float32)
    v = np.arange(BLOCK_SIZE, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing="ij")

    # Mask: keep coefficients where max(u, v) is in [low_cutoff, high_cutoff]
    freq = np.maximum(U, V)
    mask = ((freq >= low_cutoff) & (freq <= high_cutoff)).astype(np.float32)

    y = (img.y_coeffs.astype(np.float32) * mask).astype(np.int16)

    cb = None
    cr = None
    if img.cb_coeffs is not None:
        cb = (img.cb_coeffs.astype(np.float32) * mask).astype(np.int16)
        cr = (img.cr_coeffs.astype(np.float32) * mask).astype(np.int16)

    return DCTImage(
        y_coeffs=y, cb_coeffs=cb, cr_coeffs=cr,
        quant_tables=img.quant_tables,
        width=img.width, height=img.height,
        comp_info=img.comp_info,
        source_path=img._source_path,
    )


def unsharp_mask(
    img: DCTImage,
    sigma: float = 2.0,
    amount: float = 1.5,
) -> DCTImage:
    """Unsharp mask in DCT domain.

    Sharpens by: result = original + amount * (original - blurred)
    In frequency domain: weight(u,v) = 1 + amount * (1 - gaussian(u,v))

    Parameters
    ----------
    img : DCTImage
        Input image.
    sigma : float
        Gaussian sigma for the blur component.
    amount : float
        Sharpening strength. 0 = no change, 1.5 = typical.

    Returns
    -------
    DCTImage
    """
    blur_env = gaussian_envelope(max(sigma, 0.01))

    # unsharp weight = 1 + amount * (1 - blur_envelope)
    weights = (1.0 + amount * (1.0 - blur_env)).astype(np.float32)
    weights[0, 0] = 1.0  # preserve DC

    y = np.round(img.y_coeffs.astype(np.float32) * weights).astype(np.int16)

    cb = None
    cr = None
    if img.cb_coeffs is not None:
        cb = img.cb_coeffs.copy()
        cr = img.cr_coeffs.copy()

    return DCTImage(
        y_coeffs=y, cb_coeffs=cb, cr_coeffs=cr,
        quant_tables=img.quant_tables,
        width=img.width, height=img.height,
        comp_info=img.comp_info,
        source_path=img._source_path,
    )
