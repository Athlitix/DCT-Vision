"""Frequency-domain denoising and JPEG deblocking.

Wiener filter: optimal linear filter for noise reduction in frequency domain.
JPEG deblocking: attenuate high-frequency quantization artifacts.
"""

from __future__ import annotations

import numpy as np

from dct_vision.core.dct_image import DCTImage
from dct_vision.utils.constants import BLOCK_SIZE


def wiener_denoise(
    img: DCTImage,
    noise_variance: float = 5.0,
) -> DCTImage:
    """Wiener denoising in DCT domain.

    The Wiener filter is the optimal linear filter for minimizing MSE
    in the presence of additive noise. In frequency domain:

        W(u,v) = |S(u,v)|^2 / (|S(u,v)|^2 + noise_variance)

    where S is the signal power. We estimate signal power from the
    coefficients themselves.

    Parameters
    ----------
    img : DCTImage
        Input image (potentially noisy).
    noise_variance : float
        Estimated noise variance in coefficient space.

    Returns
    -------
    DCTImage
    """
    y = img.y_coeffs.astype(np.float32)

    # Estimate signal power per frequency position across all blocks
    signal_power = np.mean(y ** 2, axis=(0, 1))  # shape (8, 8)

    # Wiener filter: W = signal_power / (signal_power + noise_var)
    wiener = signal_power / (signal_power + noise_variance)
    wiener[0, 0] = 1.0  # preserve DC

    y_filtered = np.round(y * wiener).astype(np.int16)

    cb = img.cb_coeffs
    cr = img.cr_coeffs
    if cb is not None:
        cb_power = np.mean(cb.astype(np.float32) ** 2, axis=(0, 1))
        cb_wiener = cb_power / (cb_power + noise_variance)
        cb_wiener[0, 0] = 1.0
        cb = np.round(cb.astype(np.float32) * cb_wiener).astype(np.int16)
        cr_power = np.mean(cr.astype(np.float32) ** 2, axis=(0, 1))
        cr_wiener = cr_power / (cr_power + noise_variance)
        cr_wiener[0, 0] = 1.0
        cr = np.round(cr.astype(np.float32) * cr_wiener).astype(np.int16)

    return DCTImage(
        y_coeffs=y_filtered,
        cb_coeffs=cb,
        cr_coeffs=cr,
        quant_tables=img.quant_tables,
        width=img.width, height=img.height,
        comp_info=img.comp_info,
        source_path=img._source_path,
    )


def jpeg_deblock(
    img: DCTImage,
    strength: float = 1.0,
) -> DCTImage:
    """Reduce JPEG blocking artifacts in DCT domain.

    Blocking artifacts manifest as excess energy in high-frequency
    coefficients near the Nyquist frequency (positions 6-7 in the 8x8
    block). This filter attenuates those frequencies proportional to
    the quantization table values -- heavier quantization means more
    artifacts to remove.

    Parameters
    ----------
    img : DCTImage
        Input image (typically low-quality JPEG).
    strength : float
        Deblocking strength. 1.0 = standard, 2.0 = aggressive.

    Returns
    -------
    DCTImage
    """
    u = np.arange(BLOCK_SIZE, dtype=np.float32)
    v = np.arange(BLOCK_SIZE, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing="ij")

    # Attenuation increases with frequency -- target the highest frequencies
    # where blocking artifacts concentrate
    freq = np.sqrt(U**2 + V**2) / np.sqrt(2 * (BLOCK_SIZE - 1)**2)
    # Smooth rolloff starting at freq > 0.5
    attenuation = np.where(
        freq > 0.5,
        1.0 - strength * (freq - 0.5) / 0.5,
        1.0,
    )
    attenuation = np.clip(attenuation, 0.0, 1.0).astype(np.float32)
    attenuation[0, 0] = 1.0  # preserve DC

    y = np.round(img.y_coeffs.astype(np.float32) * attenuation).astype(np.int16)

    cb = img.cb_coeffs
    cr = img.cr_coeffs
    if cb is not None:
        cb = np.round(cb.astype(np.float32) * attenuation).astype(np.int16)
        cr = np.round(cr.astype(np.float32) * attenuation).astype(np.int16)

    return DCTImage(
        y_coeffs=y,
        cb_coeffs=cb,
        cr_coeffs=cr,
        quant_tables=img.quant_tables,
        width=img.width, height=img.height,
        comp_info=img.comp_info,
        source_path=img._source_path,
    )
