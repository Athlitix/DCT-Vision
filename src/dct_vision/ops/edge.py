"""Edge detection in DCT domain.

Laplacian: multiply coefficient at (u,v) by -(u² + v²)
Gradient: magnitude of horizontal (u) and vertical (v) derivatives
"""

from __future__ import annotations

import numpy as np

from dct_vision.core.dct_image import DCTImage
from dct_vision.utils.constants import BLOCK_SIZE


def _laplacian_weights() -> np.ndarray:
    """Generate Laplacian weight matrix for DCT coefficients.

    In frequency domain, the Laplacian ∇²f maps to -(u² + v²) · F(u,v).
    """
    u = np.arange(BLOCK_SIZE, dtype=np.float32)
    v = np.arange(BLOCK_SIZE, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing="ij")
    weights = -(U**2 + V**2)
    # Normalize to prevent overflow
    max_val = abs(weights).max()
    if max_val > 0:
        weights /= max_val
    return weights.astype(np.float32)


def _gradient_weights() -> tuple[np.ndarray, np.ndarray]:
    """Generate gradient weight matrices (horizontal and vertical).

    First derivative: multiply by frequency index u (horizontal) or v (vertical).
    """
    u = np.arange(BLOCK_SIZE, dtype=np.float32)
    v = np.arange(BLOCK_SIZE, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing="ij")

    max_val = BLOCK_SIZE - 1
    h_weights = (U / max_val).astype(np.float32)
    v_weights = (V / max_val).astype(np.float32)

    return h_weights, v_weights


def detect_edges(img: DCTImage, method: str = "laplacian") -> DCTImage:
    """Detect edges in DCT domain.

    Returns a grayscale edge map as a DCTImage.

    Parameters
    ----------
    img : DCTImage
        Input image.
    method : str
        'laplacian' or 'gradient'.

    Returns
    -------
    DCTImage
        Grayscale edge map.

    Raises
    ------
    ValueError
        If method is not recognized.
    """
    if method not in ("laplacian", "gradient"):
        raise ValueError(f"method must be 'laplacian' or 'gradient', got '{method}'")

    # Work on luma channel only
    y = img.y_coeffs.astype(np.float32)

    if method == "laplacian":
        weights = _laplacian_weights()
        edge_coeffs = np.round(y * weights).astype(np.int16)
    else:
        h_weights, v_weights = _gradient_weights()
        h_edge = y * h_weights
        v_edge = y * v_weights
        # Gradient magnitude
        magnitude = np.sqrt(h_edge**2 + v_edge**2)
        edge_coeffs = np.round(magnitude).astype(np.int16)

    # Return as grayscale DCTImage
    qtable = img.quant_tables[0]

    return DCTImage(
        y_coeffs=edge_coeffs,
        cb_coeffs=None,
        cr_coeffs=None,
        quant_tables=[qtable],
        width=img.width,
        height=img.height,
        comp_info=[{"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": 0}],
    )
