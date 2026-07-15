"""JPEG double-compression detection from DCT coefficient statistics.

When a JPEG is decoded and re-compressed, the histogram of each AC coefficient
develops periodic peaks and gaps (the "double quantization" effect): values
snapped to multiples of the first quantization step, then re-snapped to the
second, cluster on a comb. A singly compressed image has a smooth (roughly
Laplacian) coefficient histogram, so its histogram varies slowly; a comb makes
the histogram's second difference large.

Limitation
----------
The effect is detectable when an earlier compression was as coarse or coarser
than the final one (the common "recompressed / re-saved at higher quality"
case). If the *final* compression is the coarsest, its quantization dominates
and erases the earlier structure -- double compression is then largely
undetectable from the final coefficients. This matches the JPEG forensics
literature.
"""

from __future__ import annotations

import numpy as np

from dct_vision.core.dct_image import DCTImage

# Low-frequency AC positions (skip DC at (0,0)); these carry most signal energy.
_AC_POSITIONS = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)]
_HIST_RANGE = 48  # histogram coefficient values in [-48, 48]


def _comb_score(values: np.ndarray) -> float:
    """Comb-ness of a coefficient histogram (0 = smooth Laplacian).

    Measured as the total absolute second difference of the normalized
    histogram relative to its peak: a smooth histogram has a small second
    difference, a double-quantization comb has a large one.
    """
    v = values[np.abs(values) <= _HIST_RANGE].astype(np.int64)
    if v.size < 64:
        return 0.0
    hist = np.bincount(v + _HIST_RANGE, minlength=2 * _HIST_RANGE + 1).astype(np.float64)
    total = hist.sum()
    if total < 1:
        return 0.0
    hist /= total
    d2 = np.abs(np.diff(hist, 2))
    return float(d2.sum() / (hist.max() + 1e-9))


def detect_double_compression(img: DCTImage, threshold: float = 12.0) -> dict:
    """Estimate whether a JPEG has been compressed more than once.

    Parameters
    ----------
    img : DCTImage
        Input image (loaded from the JPEG under test).
    threshold : float
        Comb score above which the image is flagged as double-compressed. The
        default (12.0) is a heuristic and is somewhat quality-dependent (higher
        final quality raises baseline scores); the returned ``score`` lets
        callers pick their own operating point.

    Returns
    -------
    dict
        ``{"score": float, "is_double_compressed": bool, "per_position": {...}}``
        where ``score`` is the mean comb score over the analysed AC positions.
    """
    y = img.y_coeffs

    per_pos = {}
    scores = []
    for (u, v) in _AC_POSITIONS:
        vals = y[:, :, u, v].reshape(-1)
        s = _comb_score(vals)
        per_pos[f"{u},{v}"] = round(s, 3)
        scores.append(s)

    score = float(np.mean(scores)) if scores else 0.0
    return {
        "score": round(score, 3),
        "is_double_compressed": score > threshold,
        "per_position": per_pos,
    }
