"""Y/Cb/Cr channel handling and subsampling utilities."""

import math

VALID_SUBSAMPLING = {"4:4:4", "4:2:2", "4:2:0"}


def validate_subsampling(mode: str) -> None:
    """Validate a chroma subsampling mode string.

    Raises
    ------
    ValueError
        If mode is not one of '4:4:4', '4:2:2', '4:2:0'.
    """
    if mode not in VALID_SUBSAMPLING:
        raise ValueError(
            f"Unsupported subsampling mode '{mode}'. "
            f"Must be one of {sorted(VALID_SUBSAMPLING)}."
        )


def chroma_dimensions(
    luma_height: int, luma_width: int, subsampling: str
) -> tuple[int, int]:
    """Compute chroma channel dimensions given luma dimensions and subsampling.

    Parameters
    ----------
    luma_height : int
        Height of the luminance channel.
    luma_width : int
        Width of the luminance channel.
    subsampling : str
        Chroma subsampling mode ('4:4:4', '4:2:2', '4:2:0').

    Returns
    -------
    tuple[int, int]
        (chroma_height, chroma_width).
    """
    validate_subsampling(subsampling)

    if subsampling == "4:4:4":
        return luma_height, luma_width
    elif subsampling == "4:2:2":
        return luma_height, math.ceil(luma_width / 2)
    else:  # 4:2:0
        return math.ceil(luma_height / 2), math.ceil(luma_width / 2)
