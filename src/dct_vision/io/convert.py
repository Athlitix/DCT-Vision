"""Format conversion: non-JPEG images to DCTImage."""

import logging
import numpy as np
from PIL import Image

from dct_vision.core.dct_image import DCTImage
from dct_vision.exceptions import UnsupportedFormatError

logger = logging.getLogger(__name__)


def convert_to_dct(path: str, quality: int = 85) -> DCTImage:
    """Convert any Pillow-readable image to a DCTImage.

    Pipeline: read pixels -> RGB/grayscale -> DCTImage.from_array().

    Parameters
    ----------
    path : str
        Path to image file (PNG, BMP, TIFF, etc.).
    quality : int
        JPEG quality factor for quantization (1-100).

    Returns
    -------
    DCTImage
    """
    try:
        img = Image.open(path)
    except Exception as e:
        raise UnsupportedFormatError(f"Cannot open image: {path}: {e}")

    if img.mode == "RGBA":
        logger.warning("RGBA image detected — dropping alpha channel: %s", path)
        img = img.convert("RGB")
    elif img.mode == "L":
        pass  # grayscale, keep as-is
    elif img.mode != "RGB":
        img = img.convert("RGB")

    pixels = np.array(img, dtype=np.uint8)
    return DCTImage.from_array(pixels, quality=quality)
