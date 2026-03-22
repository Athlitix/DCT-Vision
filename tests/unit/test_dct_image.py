"""Tests for DCTImage core data structure."""

import numpy as np
import pytest
from PIL import Image

from dct_vision.core.dct_image import DCTImage
from dct_vision.exceptions import InvalidImageError


class TestDCTImageFromFile:
    def test_load_color_jpeg(self, sample_jpeg):
        img = DCTImage.from_file(str(sample_jpeg))
        assert img.width == 256
        assert img.height == 256
        assert img.num_components == 3
        assert img.y_coeffs is not None
        assert img.cb_coeffs is not None
        assert img.cr_coeffs is not None

    def test_load_grayscale_jpeg(self, grayscale_jpeg):
        img = DCTImage.from_file(str(grayscale_jpeg))
        assert img.num_components == 1
        assert img.y_coeffs is not None
        assert img.cb_coeffs is None
        assert img.cr_coeffs is None

    def test_coefficient_shapes(self, sample_jpeg):
        img = DCTImage.from_file(str(sample_jpeg))
        assert img.y_coeffs.ndim == 4
        assert img.y_coeffs.shape[2] == 8
        assert img.y_coeffs.shape[3] == 8

    def test_has_quant_tables(self, sample_jpeg):
        img = DCTImage.from_file(str(sample_jpeg))
        assert len(img.quant_tables) >= 1
        assert img.quant_tables[0].shape == (8, 8)

    def test_nonexistent_raises(self, tmp_path):
        with pytest.raises(InvalidImageError):
            DCTImage.from_file(str(tmp_path / "nope.jpg"))


class TestDCTImageFromArray:
    def test_from_rgb_array(self):
        pixels = np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img = DCTImage.from_array(pixels, quality=85)
        assert img.width == 64
        assert img.height == 64
        assert img.num_components == 3

    def test_from_grayscale_array(self):
        pixels = np.random.RandomState(42).randint(0, 256, (64, 64), dtype=np.uint8)
        img = DCTImage.from_array(pixels, quality=85)
        assert img.num_components == 1

    def test_non_multiple_of_8(self):
        pixels = np.random.RandomState(42).randint(0, 256, (100, 77, 3), dtype=np.uint8)
        img = DCTImage.from_array(pixels, quality=85)
        assert img.width == 77
        assert img.height == 100


class TestDCTImageToPixels:
    def test_returns_uint8_array(self, sample_jpeg):
        img = DCTImage.from_file(str(sample_jpeg))
        pixels = img.to_pixels()
        assert pixels.dtype == np.uint8
        assert pixels.shape == (256, 256, 3)

    def test_pixel_range(self, sample_jpeg):
        img = DCTImage.from_file(str(sample_jpeg))
        pixels = img.to_pixels()
        assert pixels.min() >= 0
        assert pixels.max() <= 255

    def test_grayscale_to_pixels(self, grayscale_jpeg):
        img = DCTImage.from_file(str(grayscale_jpeg))
        pixels = img.to_pixels()
        assert pixels.ndim == 2
        assert pixels.dtype == np.uint8


class TestDCTImageSave:
    def test_save_produces_valid_jpeg(self, sample_jpeg, tmp_path):
        img = DCTImage.from_file(str(sample_jpeg))
        out_path = tmp_path / "saved.jpg"
        img.save(str(out_path))
        assert out_path.exists()
        # Verify it's a valid JPEG by opening with Pillow
        reopened = Image.open(str(out_path))
        assert reopened.size == (256, 256)

    def test_save_grayscale(self, grayscale_jpeg, tmp_path):
        img = DCTImage.from_file(str(grayscale_jpeg))
        out_path = tmp_path / "gray_saved.jpg"
        img.save(str(out_path))
        assert out_path.exists()
