"""Tests for brightness and contrast DCT-domain operations."""

import numpy as np
import pytest

from dct_vision.core.dct_image import DCTImage
from dct_vision.ops.color import adjust_brightness, adjust_contrast


class TestAdjustBrightness:
    def test_positive_offset_increases_dc(self):
        """Brightness increase should add to DC coefficient."""
        img = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        original_dc = img.y_coeffs[:, :, 0, 0].copy()
        result = adjust_brightness(img, offset=30)
        # DC coefficients should increase
        assert np.all(result.y_coeffs[:, :, 0, 0] > original_dc)

    def test_negative_offset_decreases_dc(self):
        img = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        original_dc = img.y_coeffs[:, :, 0, 0].copy()
        result = adjust_brightness(img, offset=-30)
        assert np.all(result.y_coeffs[:, :, 0, 0] < original_dc)

    def test_zero_offset_is_identity(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = adjust_brightness(img, offset=0)
        np.testing.assert_array_equal(result.y_coeffs, img.y_coeffs)

    def test_ac_coefficients_unchanged(self):
        """Brightness should only affect DC (0,0), not AC coefficients."""
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = adjust_brightness(img, offset=50)
        # AC coefficients (everything except [0,0]) should be unchanged
        np.testing.assert_array_equal(
            result.y_coeffs[:, :, 1:, :], img.y_coeffs[:, :, 1:, :]
        )
        np.testing.assert_array_equal(
            result.y_coeffs[:, :, 0, 1:], img.y_coeffs[:, :, 0, 1:]
        )

    def test_chroma_unchanged(self):
        """Brightness should only affect luma (Y), not chroma (Cb, Cr)."""
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = adjust_brightness(img, offset=50)
        np.testing.assert_array_equal(result.cb_coeffs, img.cb_coeffs)
        np.testing.assert_array_equal(result.cr_coeffs, img.cr_coeffs)

    def test_returns_new_dct_image(self):
        """Should return a new DCTImage, not modify in place."""
        img = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        result = adjust_brightness(img, offset=30)
        assert result is not img

    def test_visual_brightness_increases(self):
        """Pixel-level brightness should actually increase."""
        img = DCTImage.from_array(
            np.full((64, 64, 3), 100, dtype=np.uint8), quality=95
        )
        original_mean = img.to_pixels().mean()
        result = adjust_brightness(img, offset=30)
        result_mean = result.to_pixels().mean()
        assert result_mean > original_mean


class TestAdjustContrast:
    def test_factor_greater_than_1_boosts_ac(self):
        """Contrast > 1 should scale up AC coefficients."""
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = adjust_contrast(img, factor=2.0)
        # AC energy should increase
        original_ac_energy = np.sum(img.y_coeffs[:, :, 1:, :].astype(np.float64) ** 2)
        result_ac_energy = np.sum(result.y_coeffs[:, :, 1:, :].astype(np.float64) ** 2)
        assert result_ac_energy > original_ac_energy

    def test_factor_less_than_1_reduces_ac(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = adjust_contrast(img, factor=0.5)
        original_ac_energy = np.sum(img.y_coeffs[:, :, 1:, :].astype(np.float64) ** 2)
        result_ac_energy = np.sum(result.y_coeffs[:, :, 1:, :].astype(np.float64) ** 2)
        assert result_ac_energy < original_ac_energy

    def test_factor_1_is_identity(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = adjust_contrast(img, factor=1.0)
        np.testing.assert_array_equal(result.y_coeffs, img.y_coeffs)

    def test_dc_unchanged(self):
        """Contrast should not affect DC coefficient (block mean)."""
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = adjust_contrast(img, factor=2.0)
        np.testing.assert_array_equal(
            result.y_coeffs[:, :, 0, 0], img.y_coeffs[:, :, 0, 0]
        )

    def test_invalid_factor_raises(self):
        img = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        with pytest.raises(ValueError):
            adjust_contrast(img, factor=-1.0)
