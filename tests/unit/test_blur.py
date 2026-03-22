"""Tests for Gaussian blur in DCT domain."""

import numpy as np
import pytest

from dct_vision.core.dct_image import DCTImage
from dct_vision.ops.blur import blur, gaussian_envelope


class TestGaussianEnvelope:
    def test_shape(self):
        env = gaussian_envelope(sigma=2.0)
        assert env.shape == (8, 8)

    def test_dc_is_one(self):
        """DC component (0,0) should always be 1.0."""
        env = gaussian_envelope(sigma=2.0)
        assert env[0, 0] == pytest.approx(1.0)

    def test_high_freq_attenuated(self):
        """Higher frequencies should have smaller weights."""
        env = gaussian_envelope(sigma=2.0)
        assert env[7, 7] < env[0, 0]
        assert env[7, 7] < env[1, 1]

    def test_monotonic_decay_along_diagonal(self):
        env = gaussian_envelope(sigma=2.0)
        diag = [env[i, i] for i in range(8)]
        for i in range(1, len(diag)):
            assert diag[i] <= diag[i - 1]

    def test_small_sigma_attenuates_more(self):
        """Smaller sigma = more aggressive low-pass = more attenuation."""
        env_small = gaussian_envelope(sigma=0.5)
        env_large = gaussian_envelope(sigma=4.0)
        # High-freq component should be more attenuated with small sigma
        assert env_small[4, 4] < env_large[4, 4]

    def test_large_sigma_preserves_more(self):
        """Very large sigma should preserve nearly all frequencies."""
        env = gaussian_envelope(sigma=100.0)
        assert env[7, 7] > 0.99

    def test_dtype_float32(self):
        env = gaussian_envelope(sigma=2.0)
        assert env.dtype == np.float32

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError):
            gaussian_envelope(sigma=0.0)
        with pytest.raises(ValueError):
            gaussian_envelope(sigma=-1.0)


class TestBlur:
    def test_blur_reduces_high_freq_energy(self):
        """Blur should reduce energy in high-frequency coefficients."""
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = blur(img, sigma=2.0)

        # High-freq energy should decrease
        orig_hf = np.sum(img.y_coeffs[:, :, 4:, 4:].astype(np.float64) ** 2)
        result_hf = np.sum(result.y_coeffs[:, :, 4:, 4:].astype(np.float64) ** 2)
        assert result_hf < orig_hf

    def test_dc_preserved(self):
        """Blur should preserve DC coefficients (block mean brightness)."""
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = blur(img, sigma=2.0)
        np.testing.assert_array_equal(
            result.y_coeffs[:, :, 0, 0], img.y_coeffs[:, :, 0, 0]
        )

    def test_stronger_blur_removes_more(self):
        """Larger sigma should remove more high-frequency content."""
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=95,
        )
        light = blur(img, sigma=4.0)
        heavy = blur(img, sigma=1.0)

        # Use mid-frequency range (2:6) to avoid quantization floor at highest freqs
        light_energy = np.sum(light.y_coeffs[:, :, 2:6, 2:6].astype(np.float64) ** 2)
        heavy_energy = np.sum(heavy.y_coeffs[:, :, 2:6, 2:6].astype(np.float64) ** 2)
        assert heavy_energy < light_energy

    def test_returns_new_image(self):
        img = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        result = blur(img, sigma=2.0)
        assert result is not img

    def test_chroma_also_blurred_by_default(self):
        """By default, chroma channels should also be blurred."""
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = blur(img, sigma=2.0)
        # Cb high-freq should decrease
        orig_cb_hf = np.sum(img.cb_coeffs[:, :, 4:, 4:].astype(np.float64) ** 2)
        result_cb_hf = np.sum(result.cb_coeffs[:, :, 4:, 4:].astype(np.float64) ** 2)
        assert result_cb_hf <= orig_cb_hf

    def test_luma_only_mode(self):
        """channels='luma' should only blur Y, leave Cb/Cr unchanged."""
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = blur(img, sigma=2.0, channels="luma")
        np.testing.assert_array_equal(result.cb_coeffs, img.cb_coeffs)
        np.testing.assert_array_equal(result.cr_coeffs, img.cr_coeffs)

    def test_invalid_sigma_raises(self):
        img = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        with pytest.raises(ValueError):
            blur(img, sigma=0.0)

    def test_blurred_image_saveable(self, tmp_path):
        """Blurred image should save to a valid JPEG."""
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = blur(img, sigma=2.0)
        out = tmp_path / "blurred.jpg"
        result.save(str(out))
        assert out.exists()
