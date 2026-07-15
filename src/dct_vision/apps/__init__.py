"""Practical DCT-native applications built on the core operations.

- dedup: perceptual-hash duplicate detection over image folders.
- forensics: JPEG double-compression detection from coefficient histograms.
- thumbnail: instant DC-only thumbnail generation (no IDCT).
"""

from dct_vision.apps.dedup import find_duplicates
from dct_vision.apps.forensics import detect_double_compression
from dct_vision.apps.thumbnail import dc_thumbnail

__all__ = ["find_duplicates", "detect_double_compression", "dc_thumbnail"]
