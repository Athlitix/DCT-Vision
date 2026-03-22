"""Tests for channel utilities."""

import pytest
from dct_vision.core.channel import chroma_dimensions, validate_subsampling


class TestChromaDimensions:
    def test_444(self):
        ch, cw = chroma_dimensions(256, 256, "4:4:4")
        assert ch == 256
        assert cw == 256

    def test_422(self):
        ch, cw = chroma_dimensions(256, 256, "4:2:2")
        assert ch == 256
        assert cw == 128

    def test_420(self):
        ch, cw = chroma_dimensions(256, 256, "4:2:0")
        assert ch == 128
        assert cw == 128

    def test_odd_luma_420(self):
        """Chroma dimensions should round up for odd luma sizes."""
        ch, cw = chroma_dimensions(255, 255, "4:2:0")
        assert ch == 128  # ceil(255/2)
        assert cw == 128


class TestValidateSubsampling:
    def test_valid_modes(self):
        for mode in ["4:4:4", "4:2:2", "4:2:0"]:
            validate_subsampling(mode)  # Should not raise

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            validate_subsampling("4:1:1")
        with pytest.raises(ValueError):
            validate_subsampling("invalid")
