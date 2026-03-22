"""Custom exception hierarchy for dct-vision."""


class DCTVisionError(Exception):
    """Base exception for all dct-vision errors."""


class InvalidImageError(DCTVisionError):
    """Image cannot be loaded or is corrupt."""


class UnsupportedFormatError(DCTVisionError):
    """Image format not supported for direct DCT extraction."""


class BlockBoundaryError(DCTVisionError):
    """Operation failed due to cross-block boundary issues."""


class LibjpegError(DCTVisionError):
    """Error from underlying libjpeg library."""
