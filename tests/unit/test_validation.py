"""Tests for I/O validation utilities."""

import pytest
from pathlib import Path

from dct_vision.io.validation import validate_jpeg_file, detect_format
from dct_vision.exceptions import InvalidImageError, UnsupportedFormatError


class TestValidateJpegFile:
    def test_valid_jpeg(self, sample_jpeg):
        validate_jpeg_file(str(sample_jpeg))  # Should not raise

    def test_nonexistent_raises(self, tmp_path):
        with pytest.raises(InvalidImageError, match="not found"):
            validate_jpeg_file(str(tmp_path / "nope.jpg"))

    def test_non_jpeg_raises(self, sample_png):
        with pytest.raises(InvalidImageError, match="Not a JPEG"):
            validate_jpeg_file(str(sample_png))

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.jpg"
        empty.write_bytes(b"")
        with pytest.raises(InvalidImageError):
            validate_jpeg_file(str(empty))


class TestDetectFormat:
    def test_jpeg(self, sample_jpeg):
        assert detect_format(str(sample_jpeg)) == "jpeg"

    def test_png(self, sample_png):
        assert detect_format(str(sample_png)) == "png"

    def test_bmp(self, sample_bmp):
        assert detect_format(str(sample_bmp)) == "bmp"

    def test_unknown_raises(self, tmp_path):
        unknown = tmp_path / "mystery.bin"
        unknown.write_bytes(b"\x00\x00\x00\x00")
        with pytest.raises(UnsupportedFormatError):
            detect_format(str(unknown))
