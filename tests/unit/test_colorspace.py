"""Tests for RGB <-> YCbCr colorspace conversion."""

import numpy as np
import pytest
from dct_vision.math.colorspace import rgb_to_ycbcr, ycbcr_to_rgb


class TestRGBToYCbCr:
    def test_black(self):
        rgb = np.zeros((1, 1, 3), dtype=np.float32)
        ycbcr = rgb_to_ycbcr(rgb)
        assert ycbcr.shape == (1, 1, 3)
        assert ycbcr[0, 0, 0] == pytest.approx(0.0, abs=1.0)       # Y
        assert ycbcr[0, 0, 1] == pytest.approx(128.0, abs=1.0)     # Cb
        assert ycbcr[0, 0, 2] == pytest.approx(128.0, abs=1.0)     # Cr

    def test_white(self):
        rgb = np.full((1, 1, 3), 255.0, dtype=np.float32)
        ycbcr = rgb_to_ycbcr(rgb)
        assert ycbcr[0, 0, 0] == pytest.approx(255.0, abs=1.0)     # Y
        assert ycbcr[0, 0, 1] == pytest.approx(128.0, abs=1.0)     # Cb
        assert ycbcr[0, 0, 2] == pytest.approx(128.0, abs=1.0)     # Cr

    def test_pure_red(self):
        rgb = np.array([[[255.0, 0.0, 0.0]]], dtype=np.float32)
        ycbcr = rgb_to_ycbcr(rgb)
        # ITU-R BT.601: Y = 0.299*R
        assert ycbcr[0, 0, 0] == pytest.approx(76.245, abs=1.0)

    def test_output_dtype_float32(self):
        rgb = np.ones((4, 4, 3), dtype=np.float32) * 128
        ycbcr = rgb_to_ycbcr(rgb)
        assert ycbcr.dtype == np.float32

    def test_batch_shape_preserved(self):
        rgb = np.random.rand(16, 32, 3).astype(np.float32) * 255
        ycbcr = rgb_to_ycbcr(rgb)
        assert ycbcr.shape == (16, 32, 3)


class TestYCbCrToRGB:
    def test_roundtrip(self):
        """RGB -> YCbCr -> RGB should be near-identity."""
        rgb = np.random.RandomState(42).rand(8, 8, 3).astype(np.float32) * 255
        recovered = ycbcr_to_rgb(rgb_to_ycbcr(rgb))
        np.testing.assert_allclose(recovered, rgb, atol=0.5)

    def test_output_clamped_0_255(self):
        """Output should be clamped to [0, 255]."""
        # Extreme YCbCr values that would produce out-of-range RGB
        ycbcr = np.array([[[255.0, 0.0, 255.0]]], dtype=np.float32)
        rgb = ycbcr_to_rgb(ycbcr)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 255.0

    def test_output_dtype_float32(self):
        ycbcr = np.ones((4, 4, 3), dtype=np.float32) * 128
        rgb = ycbcr_to_rgb(ycbcr)
        assert rgb.dtype == np.float32
