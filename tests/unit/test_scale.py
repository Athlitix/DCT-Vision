"""Tests for downscaling via frequency truncation."""

import numpy as np
import pytest

from dct_vision.core.dct_image import DCTImage
from dct_vision.ops.scale import downscale


class TestDownscale:
    def test_halves_dimensions(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (128, 128, 3), dtype=np.uint8),
            quality=85,
        )
        result = downscale(img, factor=2)
        assert result.width == 64
        assert result.height == 64

    def test_quarter_dimensions(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (128, 128, 3), dtype=np.uint8),
            quality=85,
        )
        result = downscale(img, factor=4)
        assert result.width == 32
        assert result.height == 32

    def test_coefficient_block_count_decreases(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (128, 128, 3), dtype=np.uint8),
            quality=85,
        )
        result = downscale(img, factor=2)
        # Original: 16x16 blocks, downscaled: 8x8 blocks
        assert result.y_coeffs.shape[0] < img.y_coeffs.shape[0]
        assert result.y_coeffs.shape[1] < img.y_coeffs.shape[1]

    def test_block_size_preserved(self):
        """Each block should still be 8x8."""
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (128, 128, 3), dtype=np.uint8),
            quality=85,
        )
        result = downscale(img, factor=2)
        assert result.y_coeffs.shape[2] == 8
        assert result.y_coeffs.shape[3] == 8

    def test_returns_new_image(self):
        img = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        result = downscale(img, factor=2)
        assert result is not img

    def test_downscaled_saveable(self, tmp_path):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (128, 128, 3), dtype=np.uint8),
            quality=85,
        )
        result = downscale(img, factor=2)
        out = tmp_path / "downscaled.jpg"
        result.save(str(out))
        assert out.exists()
        from PIL import Image as PILImage
        reloaded = PILImage.open(str(out))
        assert reloaded.size == (64, 64)

    def test_invalid_factor_raises(self):
        img = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        with pytest.raises(ValueError):
            downscale(img, factor=0)
        with pytest.raises(ValueError):
            downscale(img, factor=3)  # Only powers of 2 supported

    def test_grayscale_downscale(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (128, 128), dtype=np.uint8),
            quality=85,
        )
        result = downscale(img, factor=2)
        assert result.width == 64
        assert result.height == 64
        assert result.num_components == 1
