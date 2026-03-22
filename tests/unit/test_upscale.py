"""Tests for upscaling (decode_and_apply convenience wrapper)."""

import numpy as np
import pytest
from PIL import Image

from dct_vision.core.dct_image import DCTImage
from dct_vision.ops.scale import upscale


class TestUpscale:
    def test_doubles_dimensions(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = upscale(img, factor=2)
        assert result.width == 128
        assert result.height == 128

    def test_4x_dimensions(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = upscale(img, factor=4)
        assert result.width == 256
        assert result.height == 256

    def test_returns_new_image(self):
        img = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        result = upscale(img, factor=2)
        assert result is not img

    def test_preserves_component_count(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = upscale(img, factor=2)
        assert result.num_components == 3

    def test_grayscale_upscale(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64), dtype=np.uint8),
            quality=85,
        )
        result = upscale(img, factor=2)
        assert result.width == 128
        assert result.height == 128
        assert result.num_components == 1

    def test_saveable(self, tmp_path):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = upscale(img, factor=2)
        out = tmp_path / "upscaled.jpg"
        result.save(str(out))
        assert out.exists()
        reloaded = Image.open(str(out))
        assert reloaded.size == (128, 128)

    def test_invalid_factor_raises(self):
        img = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        with pytest.raises(ValueError):
            upscale(img, factor=0)
        with pytest.raises(ValueError):
            upscale(img, factor=3)

    def test_factor_1_is_identity_dimensions(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = upscale(img, factor=1)
        assert result.width == img.width
        assert result.height == img.height
