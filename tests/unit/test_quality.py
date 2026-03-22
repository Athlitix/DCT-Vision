"""Tests for JPEG quality estimation from DCT statistics."""

import numpy as np
import pytest

from dct_vision.core.dct_image import DCTImage
from dct_vision.ops.quality import estimate_quality, dct_stats


class TestEstimateQuality:
    def test_high_quality_detected(self, sample_jpeg_q95):
        img = DCTImage.from_file(str(sample_jpeg_q95))
        q = estimate_quality(img)
        assert q > 80

    def test_low_quality_detected(self, sample_jpeg_q50):
        img = DCTImage.from_file(str(sample_jpeg_q50))
        q = estimate_quality(img)
        assert q < 70

    def test_higher_quality_gives_higher_estimate(self, sample_jpeg_q50, sample_jpeg_q95):
        img_low = DCTImage.from_file(str(sample_jpeg_q50))
        img_high = DCTImage.from_file(str(sample_jpeg_q95))
        q_low = estimate_quality(img_low)
        q_high = estimate_quality(img_high)
        assert q_high > q_low

    def test_returns_int_in_range(self, sample_jpeg):
        img = DCTImage.from_file(str(sample_jpeg))
        q = estimate_quality(img)
        assert isinstance(q, int)
        assert 1 <= q <= 100


class TestDCTStats:
    def test_returns_dict(self, sample_jpeg):
        img = DCTImage.from_file(str(sample_jpeg))
        stats = dct_stats(img)
        assert "dc_mean" in stats
        assert "dc_std" in stats
        assert "ac_energy" in stats
        assert "num_nonzero_ac" in stats
        assert "total_ac" in stats

    def test_flat_image_low_ac_energy(self):
        flat = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        stats = dct_stats(flat)
        assert stats["ac_energy"] < 100

    def test_noisy_image_high_ac_energy(self):
        noisy = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=95,
        )
        stats = dct_stats(noisy)
        assert stats["ac_energy"] > 100
