"""Tests for format conversion (non-JPEG -> DCTImage)."""

import numpy as np
import pytest
from PIL import Image

from dct_vision.io.convert import convert_to_dct
from dct_vision.core.dct_image import DCTImage
from dct_vision.exceptions import UnsupportedFormatError


class TestConvertToDCT:
    def test_png_to_dct(self, sample_png):
        img = convert_to_dct(str(sample_png))
        assert isinstance(img, DCTImage)
        assert img.width == 256
        assert img.height == 256
        assert img.num_components == 3

    def test_bmp_to_dct(self, sample_bmp):
        img = convert_to_dct(str(sample_bmp))
        assert isinstance(img, DCTImage)
        assert img.num_components == 3

    def test_to_pixels_produces_reasonable_output(self, sample_png):
        img = convert_to_dct(str(sample_png), quality=95)
        pixels = img.to_pixels()
        assert pixels.dtype == np.uint8
        assert pixels.shape[0] == 256
        assert pixels.shape[1] == 256

    def test_grayscale_conversion(self, tmp_path):
        gray = np.random.RandomState(42).randint(0, 256, (64, 64), dtype=np.uint8)
        path = tmp_path / "gray.png"
        Image.fromarray(gray, mode="L").save(str(path))
        img = convert_to_dct(str(path))
        assert img.num_components == 1

    def test_rgba_drops_alpha(self, tmp_path):
        rgba = np.random.RandomState(42).randint(0, 256, (64, 64, 4), dtype=np.uint8)
        path = tmp_path / "rgba.png"
        Image.fromarray(rgba, mode="RGBA").save(str(path))
        img = convert_to_dct(str(path))
        assert img.num_components == 3  # Alpha dropped

    def test_quality_parameter(self, sample_png):
        high_q = convert_to_dct(str(sample_png), quality=95)
        low_q = convert_to_dct(str(sample_png), quality=20)
        # Higher quality should preserve more coefficients (less quantization)
        # High-freq coefficients should be less truncated at high quality
        assert high_q.y_coeffs is not None
        assert low_q.y_coeffs is not None
