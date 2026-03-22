"""Tests for photo editing ops: vignette, sepia, grayscale, posterize, solarize."""

import numpy as np
import pytest

from dct_vision.core.dct_image import DCTImage
from dct_vision.ops.photo import (
    vignette,
    sepia,
    grayscale,
    posterize,
    solarize,
)


def _make_img(seed=42, size=64):
    px = np.random.RandomState(seed).randint(0, 256, (size, size, 3), dtype=np.uint8)
    return DCTImage.from_array(px, quality=95)


class TestVignette:
    def test_corners_darker(self):
        img = _make_img(size=128)
        result = vignette(img, strength=1.0)
        result_px = result.to_pixels()
        center_mean = result_px[48:80, 48:80].astype(float).mean()
        corner_mean = result_px[:16, :16].astype(float).mean()
        assert center_mean > corner_mean

    def test_strength_0_is_identity(self):
        img = _make_img()
        result = vignette(img, strength=0.0)
        np.testing.assert_array_equal(result.y_coeffs, img.y_coeffs)

    def test_dimensions_preserved(self):
        img = _make_img()
        result = vignette(img, strength=1.0)
        assert result.width == img.width


class TestSepia:
    def test_returns_color_image(self):
        img = _make_img()
        result = sepia(img)
        assert result.num_components == 3

    def test_chroma_modified(self):
        img = _make_img()
        result = sepia(img)
        assert not np.array_equal(result.cb_coeffs, img.cb_coeffs)

    def test_luma_preserved(self):
        img = _make_img()
        result = sepia(img)
        np.testing.assert_array_equal(result.y_coeffs, img.y_coeffs)


class TestGrayscale:
    def test_returns_grayscale(self):
        img = _make_img()
        result = grayscale(img)
        assert result.num_components == 1
        assert result.cb_coeffs is None
        assert result.cr_coeffs is None

    def test_luma_preserved(self):
        img = _make_img()
        result = grayscale(img)
        np.testing.assert_array_equal(result.y_coeffs, img.y_coeffs)

    def test_already_grayscale(self):
        gray_px = np.random.RandomState(42).randint(0, 256, (64, 64), dtype=np.uint8)
        img = DCTImage.from_array(gray_px, quality=85)
        result = grayscale(img)
        assert result.num_components == 1


class TestPosterize:
    def test_reduces_unique_coefficients(self):
        img = _make_img()
        result = posterize(img, levels=4)
        orig_unique = len(np.unique(img.y_coeffs))
        result_unique = len(np.unique(result.y_coeffs))
        assert result_unique <= orig_unique

    def test_dimensions_preserved(self):
        img = _make_img()
        result = posterize(img, levels=4)
        assert result.width == img.width

    def test_invalid_levels_raises(self):
        img = _make_img()
        with pytest.raises(ValueError):
            posterize(img, levels=0)


class TestSolarize:
    def test_modifies_coefficients(self):
        img = _make_img()
        result = solarize(img)
        assert not np.array_equal(result.y_coeffs, img.y_coeffs)

    def test_dimensions_preserved(self):
        img = _make_img()
        result = solarize(img)
        assert result.width == img.width

    def test_threshold_controls_effect(self):
        img = _make_img()
        r1 = solarize(img, threshold=5)
        r2 = solarize(img, threshold=50)
        assert not np.array_equal(r1.y_coeffs, r2.y_coeffs)
