"""Tests for image analysis: blur detection, noise estimation, complexity, similarity."""

import numpy as np
import pytest

from dct_vision.core.dct_image import DCTImage
from dct_vision.ops.analysis import (
    detect_blur,
    estimate_noise,
    texture_complexity,
    image_similarity,
    histogram_from_dct,
)


def _make_img(seed=42, size=64):
    px = np.random.RandomState(seed).randint(0, 256, (size, size, 3), dtype=np.uint8)
    return DCTImage.from_array(px, quality=95)


class TestDetectBlur:
    def test_returns_float(self):
        img = _make_img()
        score = detect_blur(img)
        assert isinstance(score, float)

    def test_blurry_image_higher_score(self):
        from dct_vision.ops.blur import blur
        img = _make_img()
        blurry = blur(img, sigma=4.0)
        # Higher score = more blurry
        assert detect_blur(blurry) > detect_blur(img)

    def test_sharp_image_low_score(self):
        img = _make_img()
        score = detect_blur(img)
        assert score < 0.9  # not maximally blurry

    def test_flat_image_max_blur(self):
        flat = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        score = detect_blur(flat)
        assert score > 0.8  # flat = very blurry


class TestEstimateNoise:
    def test_returns_float(self):
        img = _make_img()
        n = estimate_noise(img)
        assert isinstance(n, float)

    def test_noisy_image_higher(self):
        from dct_vision.augment.noise import gaussian_noise
        img = _make_img()
        noisy = gaussian_noise(img, sigma=20.0, seed=42)
        assert estimate_noise(noisy) > estimate_noise(img)

    def test_flat_image_low_noise(self):
        flat = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        assert estimate_noise(flat) < 5.0


class TestTextureComplexity:
    def test_returns_float(self):
        img = _make_img()
        c = texture_complexity(img)
        assert isinstance(c, float)
        assert 0.0 <= c <= 1.0

    def test_textured_higher_than_flat(self):
        textured = _make_img()
        flat = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        assert texture_complexity(textured) > texture_complexity(flat)


class TestImageSimilarity:
    def test_identical_is_1(self):
        img = _make_img()
        assert image_similarity(img, img) == pytest.approx(1.0)

    def test_similar_images_high(self):
        from dct_vision.ops.color import adjust_brightness
        img = _make_img()
        bright = adjust_brightness(img, offset=5)
        sim = image_similarity(img, bright)
        assert sim > 0.8

    def test_different_images_lower_than_similar(self):
        from dct_vision.ops.color import adjust_brightness
        img = _make_img(seed=1)
        similar = adjust_brightness(img, offset=5)
        different = _make_img(seed=99)
        sim_close = image_similarity(img, similar)
        sim_far = image_similarity(img, different)
        assert sim_close > sim_far

    def test_returns_float_in_range(self):
        img1 = _make_img(seed=1)
        img2 = _make_img(seed=2)
        sim = image_similarity(img1, img2)
        assert 0.0 <= sim <= 1.0


class TestHistogramFromDCT:
    def test_returns_array(self):
        img = _make_img()
        hist = histogram_from_dct(img)
        assert isinstance(hist, np.ndarray)

    def test_histogram_sums_to_block_count(self):
        img = _make_img()
        hist = histogram_from_dct(img)
        bh, bw = img.y_coeffs.shape[:2]
        assert hist.sum() == bh * bw

    def test_bin_count(self):
        img = _make_img()
        hist = histogram_from_dct(img, bins=32)
        assert len(hist) == 32
