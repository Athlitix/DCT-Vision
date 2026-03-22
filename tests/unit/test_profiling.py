"""Tests for profiling utilities."""

import numpy as np
import pytest

from dct_vision.utils.profiling import time_fn, measure_memory, psnr


class TestTimeFn:
    def test_returns_expected_keys(self):
        result = time_fn(lambda: sum(range(100)), warmup=1, repeats=5)
        assert "mean_ms" in result
        assert "std_ms" in result
        assert "min_ms" in result
        assert "max_ms" in result
        assert "repeats" in result

    def test_repeats_matches(self):
        result = time_fn(lambda: None, warmup=1, repeats=10)
        assert result["repeats"] == 10

    def test_mean_is_positive(self):
        result = time_fn(lambda: sum(range(1000)), warmup=1, repeats=5)
        assert result["mean_ms"] > 0

    def test_min_less_than_max(self):
        result = time_fn(lambda: sum(range(1000)), warmup=1, repeats=10)
        assert result["min_ms"] <= result["max_ms"]


class TestMeasureMemory:
    def test_returns_expected_keys(self):
        result = measure_memory(lambda: np.zeros((100, 100)))
        assert "peak_kb" in result
        assert "current_kb" in result

    def test_large_alloc_detected(self):
        result = measure_memory(lambda: np.ones((1000, 1000), dtype=np.float64))
        # 1000x1000 float64 = ~7.6MB
        assert result["peak_kb"] > 1000


class TestPSNR:
    def test_identical_images(self):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        assert psnr(img, img) == float("inf")

    def test_different_images(self):
        a = np.zeros((64, 64, 3), dtype=np.uint8)
        b = np.full((64, 64, 3), 10, dtype=np.uint8)
        result = psnr(a, b)
        assert 20 < result < 50

    def test_more_different_is_lower(self):
        a = np.zeros((64, 64, 3), dtype=np.uint8)
        b_small = np.full((64, 64, 3), 10, dtype=np.uint8)
        b_big = np.full((64, 64, 3), 100, dtype=np.uint8)
        assert psnr(a, b_big) < psnr(a, b_small)
