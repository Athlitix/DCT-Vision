"""Tests for sharpening in DCT domain."""

import numpy as np
import pytest

from dct_vision.core.dct_image import DCTImage
from dct_vision.ops.sharpen import sharpen


class TestSharpen:
    def test_increases_high_freq_energy(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=95,
        )
        result = sharpen(img, amount=2.0)
        orig_hf = np.sum(img.y_coeffs[:, :, 2:, 2:].astype(np.float64) ** 2)
        result_hf = np.sum(result.y_coeffs[:, :, 2:, 2:].astype(np.float64) ** 2)
        assert result_hf > orig_hf

    def test_dc_preserved(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = sharpen(img, amount=2.0)
        np.testing.assert_array_equal(
            result.y_coeffs[:, :, 0, 0], img.y_coeffs[:, :, 0, 0]
        )

    def test_amount_1_is_identity(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = sharpen(img, amount=1.0)
        np.testing.assert_array_equal(result.y_coeffs, img.y_coeffs)

    def test_returns_new_image(self):
        img = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        result = sharpen(img, amount=2.0)
        assert result is not img

    def test_invalid_amount_raises(self):
        img = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        with pytest.raises(ValueError):
            sharpen(img, amount=0.0)

    def test_saveable(self, tmp_path):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = sharpen(img, amount=1.5)
        out = tmp_path / "sharpened.jpg"
        result.save(str(out))
        assert out.exists()
