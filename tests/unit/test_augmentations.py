"""Tests for DCT-domain augmentations."""

import numpy as np
import pytest
from PIL import Image

from dct_vision.core.dct_image import DCTImage
from dct_vision.augment.flip import horizontal_flip, vertical_flip
from dct_vision.augment.crop import block_crop
from dct_vision.augment.jitter import brightness_jitter, contrast_jitter
from dct_vision.augment.noise import gaussian_noise


# -- Flip tests --

class TestHorizontalFlip:
    def test_dimensions_preserved(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = horizontal_flip(img)
        assert result.width == img.width
        assert result.height == img.height

    def test_double_flip_is_identity(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = horizontal_flip(horizontal_flip(img))
        np.testing.assert_array_equal(result.y_coeffs, img.y_coeffs)

    def test_pixels_actually_flipped(self):
        """Visual check: left-right flip at pixel level."""
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        pixels[:, :32, :] = 200  # Left half bright
        img = DCTImage.from_array(pixels, quality=95)
        result = horizontal_flip(img)
        result_px = result.to_pixels()
        # Right half should now be brighter than left
        assert result_px[:, 32:, :].mean() > result_px[:, :32, :].mean()

    def test_returns_new_image(self):
        img = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        result = horizontal_flip(img)
        assert result is not img


class TestVerticalFlip:
    def test_dimensions_preserved(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = vertical_flip(img)
        assert result.width == img.width
        assert result.height == img.height

    def test_double_flip_is_identity(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = vertical_flip(vertical_flip(img))
        np.testing.assert_array_equal(result.y_coeffs, img.y_coeffs)

    def test_pixels_actually_flipped(self):
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        pixels[:32, :, :] = 200  # Top half bright
        img = DCTImage.from_array(pixels, quality=95)
        result = vertical_flip(img)
        result_px = result.to_pixels()
        # Bottom half should now be brighter
        assert result_px[32:, :, :].mean() > result_px[:32, :, :].mean()


# -- Crop tests --

class TestBlockCrop:
    def test_crop_dimensions(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (128, 128, 3), dtype=np.uint8),
            quality=85,
        )
        result = block_crop(img, block_row=0, block_col=0, block_rows=4, block_cols=4)
        assert result.width == 32
        assert result.height == 32

    def test_crop_coefficient_shape(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (128, 128, 3), dtype=np.uint8),
            quality=85,
        )
        result = block_crop(img, block_row=2, block_col=2, block_rows=4, block_cols=4)
        assert result.y_coeffs.shape[0] == 4
        assert result.y_coeffs.shape[1] == 4

    def test_crop_out_of_bounds_raises(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        with pytest.raises(ValueError):
            block_crop(img, block_row=0, block_col=0, block_rows=100, block_cols=4)

    def test_crop_saveable(self, tmp_path):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (128, 128, 3), dtype=np.uint8),
            quality=85,
        )
        result = block_crop(img, block_row=0, block_col=0, block_rows=8, block_cols=8)
        out = tmp_path / "cropped.jpg"
        result.save(str(out))
        assert out.exists()


# -- Jitter tests --

class TestBrightnessJitter:
    def test_output_differs_from_input(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = brightness_jitter(img, max_offset=50, seed=42)
        assert not np.array_equal(result.y_coeffs, img.y_coeffs)

    def test_different_seeds_give_different_results(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        r1 = brightness_jitter(img, max_offset=50, seed=1)
        r2 = brightness_jitter(img, max_offset=50, seed=2)
        assert not np.array_equal(r1.y_coeffs, r2.y_coeffs)

    def test_zero_jitter_is_identity(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = brightness_jitter(img, max_offset=0, seed=42)
        np.testing.assert_array_equal(result.y_coeffs, img.y_coeffs)


class TestContrastJitter:
    def test_output_differs_from_input(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = contrast_jitter(img, max_factor=0.5, seed=42)
        assert not np.array_equal(result.y_coeffs, img.y_coeffs)

    def test_zero_jitter_is_identity(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = contrast_jitter(img, max_factor=0.0, seed=42)
        np.testing.assert_array_equal(result.y_coeffs, img.y_coeffs)


# -- Noise tests --

class TestGaussianNoise:
    def test_adds_noise_to_high_freq(self):
        img = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        result = gaussian_noise(img, sigma=5.0, seed=42)
        # High-freq coefficients should have changed
        hf_energy_before = np.sum(img.y_coeffs[:, :, 2:, 2:].astype(np.float64) ** 2)
        hf_energy_after = np.sum(result.y_coeffs[:, :, 2:, 2:].astype(np.float64) ** 2)
        assert hf_energy_after > hf_energy_before

    def test_dc_preserved(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = gaussian_noise(img, sigma=5.0, seed=42)
        np.testing.assert_array_equal(
            result.y_coeffs[:, :, 0, 0], img.y_coeffs[:, :, 0, 0]
        )

    def test_zero_sigma_is_identity(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = gaussian_noise(img, sigma=0.0, seed=42)
        np.testing.assert_array_equal(result.y_coeffs, img.y_coeffs)

    def test_deterministic_with_seed(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        r1 = gaussian_noise(img, sigma=5.0, seed=99)
        r2 = gaussian_noise(img, sigma=5.0, seed=99)
        np.testing.assert_array_equal(r1.y_coeffs, r2.y_coeffs)

    def test_saveable(self, tmp_path):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = gaussian_noise(img, sigma=3.0, seed=42)
        out = tmp_path / "noisy.jpg"
        result.save(str(out))
        assert out.exists()
