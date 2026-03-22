"""Tests for advanced operations: color temp, saturation, pHash, Wiener, deblocking."""

import numpy as np
import pytest

from dct_vision.core.dct_image import DCTImage


def _make_img(seed=42, size=64):
    px = np.random.RandomState(seed).randint(0, 256, (size, size, 3), dtype=np.uint8)
    return DCTImage.from_array(px, quality=95)


# -- Color temperature --

class TestColorTemperature:
    def test_warm_shifts_cr(self):
        from dct_vision.ops.color import adjust_color_temperature
        img = _make_img()
        warm = adjust_color_temperature(img, shift=30)
        # Warming shifts Cr (red) DC up
        assert warm.cr_coeffs[:, :, 0, 0].mean() > img.cr_coeffs[:, :, 0, 0].mean()

    def test_cool_shifts_cb(self):
        from dct_vision.ops.color import adjust_color_temperature
        img = _make_img()
        cool = adjust_color_temperature(img, shift=-30)
        # Cooling shifts Cb (blue) DC up
        assert cool.cb_coeffs[:, :, 0, 0].mean() > img.cb_coeffs[:, :, 0, 0].mean()

    def test_zero_shift_is_identity(self):
        from dct_vision.ops.color import adjust_color_temperature
        img = _make_img()
        result = adjust_color_temperature(img, shift=0)
        np.testing.assert_array_equal(result.y_coeffs, img.y_coeffs)
        np.testing.assert_array_equal(result.cb_coeffs, img.cb_coeffs)
        np.testing.assert_array_equal(result.cr_coeffs, img.cr_coeffs)

    def test_luma_unchanged(self):
        from dct_vision.ops.color import adjust_color_temperature
        img = _make_img()
        result = adjust_color_temperature(img, shift=50)
        np.testing.assert_array_equal(result.y_coeffs, img.y_coeffs)

    def test_grayscale_raises(self):
        from dct_vision.ops.color import adjust_color_temperature
        gray = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64), dtype=np.uint8),
            quality=85,
        )
        with pytest.raises(ValueError):
            adjust_color_temperature(gray, shift=30)


# -- Saturation --

class TestSaturation:
    def test_increase_boosts_chroma_ac(self):
        from dct_vision.ops.color import adjust_saturation
        img = _make_img()
        result = adjust_saturation(img, factor=2.0)
        orig_cb_energy = np.sum(img.cb_coeffs.astype(np.float64) ** 2)
        result_cb_energy = np.sum(result.cb_coeffs.astype(np.float64) ** 2)
        assert result_cb_energy > orig_cb_energy

    def test_decrease_reduces_chroma(self):
        from dct_vision.ops.color import adjust_saturation
        img = _make_img()
        result = adjust_saturation(img, factor=0.5)
        orig_cb_energy = np.sum(img.cb_coeffs.astype(np.float64) ** 2)
        result_cb_energy = np.sum(result.cb_coeffs.astype(np.float64) ** 2)
        assert result_cb_energy < orig_cb_energy

    def test_factor_1_is_identity(self):
        from dct_vision.ops.color import adjust_saturation
        img = _make_img()
        result = adjust_saturation(img, factor=1.0)
        np.testing.assert_array_equal(result.cb_coeffs, img.cb_coeffs)

    def test_luma_unchanged(self):
        from dct_vision.ops.color import adjust_saturation
        img = _make_img()
        result = adjust_saturation(img, factor=2.0)
        np.testing.assert_array_equal(result.y_coeffs, img.y_coeffs)

    def test_invalid_factor_raises(self):
        from dct_vision.ops.color import adjust_saturation
        img = _make_img()
        with pytest.raises(ValueError):
            adjust_saturation(img, factor=-1.0)


# -- Perceptual hash --

class TestPHash:
    def test_returns_int(self):
        from dct_vision.ops.phash import perceptual_hash
        img = _make_img()
        h = perceptual_hash(img)
        assert isinstance(h, int)

    def test_identical_images_same_hash(self):
        from dct_vision.ops.phash import perceptual_hash
        img = _make_img(seed=42)
        h1 = perceptual_hash(img)
        h2 = perceptual_hash(img)
        assert h1 == h2

    def test_different_images_different_hash(self):
        from dct_vision.ops.phash import perceptual_hash
        h1 = perceptual_hash(_make_img(seed=1))
        h2 = perceptual_hash(_make_img(seed=2))
        assert h1 != h2

    def test_similar_images_close_hamming(self):
        from dct_vision.ops.phash import perceptual_hash, hamming_distance
        from dct_vision.ops.color import adjust_brightness
        img = _make_img()
        bright = adjust_brightness(img, offset=10)
        h1 = perceptual_hash(img)
        h2 = perceptual_hash(bright)
        dist = hamming_distance(h1, h2)
        # Small brightness change should produce small Hamming distance
        assert dist < 20

    def test_very_different_images_large_hamming(self):
        from dct_vision.ops.phash import perceptual_hash, hamming_distance
        h1 = perceptual_hash(_make_img(seed=1))
        # Create a very different image
        black = DCTImage.from_array(np.zeros((64, 64, 3), dtype=np.uint8), quality=95)
        h2 = perceptual_hash(black)
        dist = hamming_distance(h1, h2)
        assert dist > 5

    def test_returns_hex_string(self):
        from dct_vision.ops.phash import perceptual_hash_hex
        img = _make_img()
        h = perceptual_hash_hex(img)
        assert isinstance(h, str)
        assert len(h) == 16  # 64 bits = 16 hex chars


# -- Wiener denoising --

class TestWienerDenoise:
    def test_reduces_high_freq_noise(self):
        from dct_vision.ops.denoise import wiener_denoise
        img = _make_img()
        # Add noise
        from dct_vision.augment.noise import gaussian_noise
        noisy = gaussian_noise(img, sigma=10.0, seed=42)
        # Denoise
        denoised = wiener_denoise(noisy, noise_variance=10.0)
        # High-freq energy should decrease
        noisy_hf = np.sum(noisy.y_coeffs[:, :, 4:, 4:].astype(np.float64) ** 2)
        denoised_hf = np.sum(denoised.y_coeffs[:, :, 4:, 4:].astype(np.float64) ** 2)
        assert denoised_hf < noisy_hf

    def test_dc_preserved(self):
        from dct_vision.ops.denoise import wiener_denoise
        img = _make_img()
        result = wiener_denoise(img, noise_variance=5.0)
        np.testing.assert_array_equal(
            result.y_coeffs[:, :, 0, 0], img.y_coeffs[:, :, 0, 0]
        )

    def test_dimensions_preserved(self):
        from dct_vision.ops.denoise import wiener_denoise
        img = _make_img()
        result = wiener_denoise(img, noise_variance=5.0)
        assert result.width == img.width
        assert result.height == img.height


# -- JPEG deblocking --

class TestDeblocking:
    def test_returns_valid_image(self):
        from dct_vision.ops.denoise import jpeg_deblock
        img = _make_img()
        result = jpeg_deblock(img)
        assert result.width == img.width
        assert result.height == img.height

    def test_reduces_block_artifacts(self):
        from dct_vision.ops.denoise import jpeg_deblock
        # Low quality = more blocking artifacts
        px = np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8)
        blocky = DCTImage.from_array(px, quality=10)
        deblocked = jpeg_deblock(blocky, strength=2.0)
        # High-freq energy (where blocking shows up) should decrease
        blocky_hf = np.sum(blocky.y_coeffs[:, :, 6:, 6:].astype(np.float64) ** 2)
        deblocked_hf = np.sum(deblocked.y_coeffs[:, :, 6:, 6:].astype(np.float64) ** 2)
        assert deblocked_hf <= blocky_hf

    def test_saveable(self, tmp_path):
        from dct_vision.ops.denoise import jpeg_deblock
        img = _make_img()
        result = jpeg_deblock(img)
        out = tmp_path / "deblocked.jpg"
        result.save(str(out))
        assert out.exists()
