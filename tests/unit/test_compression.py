"""Tests for compression ops: requantize, coefficient pruning."""

import numpy as np
import pytest

from dct_vision.core.dct_image import DCTImage
from dct_vision.ops.compression import requantize, prune_coefficients


def _make_img(seed=42, size=64):
    px = np.random.RandomState(seed).randint(0, 256, (size, size, 3), dtype=np.uint8)
    return DCTImage.from_array(px, quality=95)


class TestRequantize:
    def test_lower_quality_reduces_coefficients(self):
        img = _make_img()
        result = requantize(img, quality=20)
        # More zeros with lower quality
        orig_zeros = np.sum(img.y_coeffs == 0)
        result_zeros = np.sum(result.y_coeffs == 0)
        assert result_zeros >= orig_zeros

    def test_dimensions_preserved(self):
        img = _make_img()
        result = requantize(img, quality=50)
        assert result.width == img.width
        assert result.height == img.height

    def test_quant_tables_updated(self):
        img = _make_img()
        result = requantize(img, quality=30)
        # New quant tables should differ from original
        assert not np.array_equal(result.quant_tables[0], img.quant_tables[0])

    def test_invalid_quality_raises(self):
        img = _make_img()
        with pytest.raises(ValueError):
            requantize(img, quality=0)
        with pytest.raises(ValueError):
            requantize(img, quality=101)

    def test_saveable(self, tmp_path):
        img = _make_img()
        result = requantize(img, quality=50)
        out = tmp_path / "requantized.jpg"
        result.save(str(out))
        assert out.exists()


class TestPruneCoefficients:
    def test_increases_zeros(self):
        img = _make_img()
        result = prune_coefficients(img, threshold=2)
        orig_zeros = np.sum(img.y_coeffs == 0)
        result_zeros = np.sum(result.y_coeffs == 0)
        assert result_zeros > orig_zeros

    def test_dc_preserved(self):
        img = _make_img()
        result = prune_coefficients(img, threshold=10)
        np.testing.assert_array_equal(
            result.y_coeffs[:, :, 0, 0], img.y_coeffs[:, :, 0, 0]
        )

    def test_threshold_0_is_identity(self):
        img = _make_img()
        result = prune_coefficients(img, threshold=0)
        np.testing.assert_array_equal(result.y_coeffs, img.y_coeffs)

    def test_higher_threshold_more_zeros(self):
        img = _make_img()
        r1 = prune_coefficients(img, threshold=1)
        r2 = prune_coefficients(img, threshold=10)
        zeros_1 = np.sum(r1.y_coeffs == 0)
        zeros_2 = np.sum(r2.y_coeffs == 0)
        assert zeros_2 >= zeros_1
