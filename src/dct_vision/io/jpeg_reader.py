"""High-level JPEG reader."""

from dct_vision.core.dct_image import DCTImage
from dct_vision.io.validation import validate_jpeg_file


def read_jpeg(path: str) -> DCTImage:
    """Read a JPEG file and return a DCTImage.

    Parameters
    ----------
    path : str
        Path to JPEG file.

    Returns
    -------
    DCTImage
    """
    validate_jpeg_file(path)
    return DCTImage.from_file(path)
