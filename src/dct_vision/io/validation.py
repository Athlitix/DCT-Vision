"""File format detection and input validation."""

import os
from dct_vision.exceptions import InvalidImageError, UnsupportedFormatError

# Magic bytes for supported formats
_MAGIC = {
    b"\xff\xd8": "jpeg",
    b"\x89PNG": "png",
    b"BM": "bmp",
    b"II": "tiff",  # Little-endian TIFF
    b"MM": "tiff",  # Big-endian TIFF
}


def validate_jpeg_file(path: str) -> None:
    """Validate that a file exists and is a JPEG.

    Raises
    ------
    InvalidImageError
        If file doesn't exist, is empty, or is not a JPEG.
    """
    if not os.path.exists(path):
        raise InvalidImageError(f"File not found: {path}")

    try:
        with open(path, "rb") as f:
            magic = f.read(2)
    except IOError as e:
        raise InvalidImageError(f"Cannot read file: {path}: {e}")

    if len(magic) < 2:
        raise InvalidImageError(f"File is empty or too small: {path}")

    if magic != b"\xff\xd8":
        raise InvalidImageError(f"Not a JPEG file: {path}")


def detect_format(path: str) -> str:
    """Detect image format from file magic bytes.

    Returns
    -------
    str
        Format name: 'jpeg', 'png', 'bmp', or 'tiff'.

    Raises
    ------
    UnsupportedFormatError
        If format cannot be detected.
    """
    try:
        with open(path, "rb") as f:
            header = f.read(4)
    except IOError as e:
        raise UnsupportedFormatError(f"Cannot read file: {path}: {e}")

    # Check 4-byte signatures first, then 2-byte
    if header[:4] == b"\x89PNG":
        return "png"
    for magic, fmt in _MAGIC.items():
        if header[: len(magic)] == magic:
            return fmt

    raise UnsupportedFormatError(f"Unrecognized image format: {path}")
