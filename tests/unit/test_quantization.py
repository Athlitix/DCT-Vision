"""Tests for JPEG quantization utilities."""

import numpy as np
import pytest
from dct_vision.math.quantization import quantize, dequantize, quality_to_scale_factor, scale_quant_table
from dct_vision.utils.constants import LUMINANCE_QUANT_TABLE, CHROMINANCE_QUANT_TABLE


class TestQuantize:
    def test_basic(self):
        """Quantization divides and rounds."""
        coeffs = np.array([[160.0, 110.0], [140.0, 170.0]], dtype=np.float32)
        qtable = np.array([[16.0, 11.0], [14.0, 17.0]], dtype=np.float32)
        result = quantize(coeffs, qtable)
        expected = np.round(coeffs / qtable)
        np.testing.assert_array_equal(result, expected)

    def test_output_dtype(self):
        coeffs = np.ones((8, 8), dtype=np.float32)
        result = quantize(coeffs, LUMINANCE_QUANT_TABLE)
        assert result.dtype == np.float32

    def test_zeros_stay_zero(self):
        coeffs = np.zeros((8, 8), dtype=np.float32)
        result = quantize(coeffs, LUMINANCE_QUANT_TABLE)
        np.testing.assert_array_equal(result, 0.0)


class TestDequantize:
    def test_basic(self):
        """Dequantization multiplies."""
        qcoeffs = np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float32)
        qtable = np.array([[16.0, 11.0], [14.0, 17.0]], dtype=np.float32)
        result = dequantize(qcoeffs, qtable)
        expected = qcoeffs * qtable
        np.testing.assert_array_equal(result, expected)

    def test_quantize_dequantize_bounded_error(self):
        """Quantize -> dequantize introduces bounded error (<= qtable/2)."""
        rng = np.random.RandomState(42)
        coeffs = rng.rand(8, 8).astype(np.float32) * 1000
        qtable = LUMINANCE_QUANT_TABLE
        recovered = dequantize(quantize(coeffs, qtable), qtable)
        error = np.abs(coeffs - recovered)
        # Error per element should be at most qtable_value / 2
        np.testing.assert_array_less(error, qtable / 2 + 0.5)


class TestQualityScaling:
    def test_quality_50_is_baseline(self):
        """Quality 50 should return scale factor 1.0 (no change)."""
        sf = quality_to_scale_factor(50)
        assert sf == pytest.approx(1.0)

    def test_quality_1_is_high_quantization(self):
        """Quality 1 should produce large scale factor."""
        sf = quality_to_scale_factor(1)
        assert sf > 1.0

    def test_quality_95_is_low_quantization(self):
        """Quality 95 should produce small scale factor."""
        sf = quality_to_scale_factor(95)
        assert sf < 1.0

    def test_scale_quant_table_clamps(self):
        """Scaled table values should be >= 1."""
        scaled = scale_quant_table(LUMINANCE_QUANT_TABLE, quality=99)
        assert scaled.min() >= 1.0

    def test_quality_out_of_range_raises(self):
        with pytest.raises(ValueError):
            quality_to_scale_factor(0)
        with pytest.raises(ValueError):
            quality_to_scale_factor(101)
