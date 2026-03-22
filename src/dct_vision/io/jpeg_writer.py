"""High-level JPEG writer."""

from dct_vision.core.dct_image import DCTImage


def write_jpeg(dct_image: DCTImage, path: str, quality: int | None = None) -> None:
    """Write a DCTImage to a JPEG file.

    Parameters
    ----------
    dct_image : DCTImage
        Image to save.
    path : str
        Output file path.
    quality : int, optional
        JPEG quality. If None, uses original quantization tables.
    """
    dct_image.save(path, quality=quality)
