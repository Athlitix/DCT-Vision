"""Tests for image quality metrics (PSNR, SSIM, MSE)."""

from __future__ import annotations

import numpy as np
import pytest

from dct_vision.math.metrics import mse, psnr, ssim


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestMSE:
    def test_identical_is_zero(self, rng):
        a = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        assert mse(a, a) == 0.0

    def test_known_value(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        b = np.full((4, 4), 2, dtype=np.uint8)
        assert mse(a, b) == pytest.approx(4.0)

    def test_symmetric(self, rng):
        a = rng.integers(0, 256, (16, 16), dtype=np.uint8)
        b = rng.integers(0, 256, (16, 16), dtype=np.uint8)
        assert mse(a, b) == pytest.approx(mse(b, a))


class TestPSNR:
    def test_identical_is_inf(self, rng):
        a = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        assert psnr(a, a) == float("inf")

    def test_known_value(self):
        a = np.zeros((8, 8), dtype=np.uint8)
        b = np.full((8, 8), 1, dtype=np.uint8)
        # mse = 1 -> psnr = 10*log10(255^2) ~= 48.13
        assert psnr(a, b) == pytest.approx(48.13, abs=0.1)

    def test_higher_for_closer(self, rng):
        a = rng.integers(0, 256, (32, 32), dtype=np.uint8).astype(np.float64)
        close = np.clip(a + 1, 0, 255)
        far = np.clip(a + 20, 0, 255)
        assert psnr(a, close) > psnr(a, far)

    def test_color_image(self, rng):
        a = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        b = np.clip(a.astype(np.int16) + 3, 0, 255).astype(np.uint8)
        val = psnr(a, b)
        assert 30 < val < 50


class TestSSIM:
    def test_identical_is_one(self, rng):
        a = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        assert ssim(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_range(self, rng):
        a = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        b = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        val = ssim(a, b)
        assert -1.0 <= val <= 1.0

    def test_higher_for_closer(self, rng):
        a = rng.integers(0, 256, (64, 64), dtype=np.uint8).astype(np.float64)
        close = np.clip(a + rng.normal(0, 2, a.shape), 0, 255)
        far = np.clip(a + rng.normal(0, 40, a.shape), 0, 255)
        assert ssim(a, close) > ssim(a, far)

    def test_color_image(self, rng):
        a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        assert ssim(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_matches_skimage_if_available(self, rng):
        skimage_metrics = pytest.importorskip("skimage.metrics")
        a = rng.integers(0, 256, (128, 128), dtype=np.uint8)
        b = np.clip(a.astype(np.int16) + rng.integers(-15, 15, a.shape), 0, 255).astype(np.uint8)
        ours = ssim(a, b)
        theirs = skimage_metrics.structural_similarity(a, b, data_range=255)
        assert ours == pytest.approx(theirs, abs=0.01)
