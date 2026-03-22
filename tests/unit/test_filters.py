"""Tests for DCT-domain filters: Sobel, Scharr, box blur, emboss, band-pass, unsharp mask."""

import numpy as np
import pytest

from dct_vision.core.dct_image import DCTImage
from dct_vision.ops.filters import (
    sobel,
    scharr,
    box_blur,
    emboss,
    bandpass,
    unsharp_mask,
)


def _make_test_image(seed=42, size=64):
    pixels = np.random.RandomState(seed).randint(0, 256, (size, size, 3), dtype=np.uint8)
    return DCTImage.from_array(pixels, quality=95)


def _make_edge_image():
    """Image with edges inside blocks (not aligned to 8px grid)."""
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[:, 20:, :] = 255   # vertical edge at x=20 (inside a block)
    pixels[30:, :, :] = 200   # horizontal edge at y=30 (inside a block)
    return DCTImage.from_array(pixels, quality=95)


# -- Sobel --

class TestSobel:
    def test_horizontal_returns_grayscale(self):
        img = _make_test_image()
        result = sobel(img, direction="horizontal")
        assert result.num_components == 1

    def test_vertical_returns_grayscale(self):
        img = _make_test_image()
        result = sobel(img, direction="vertical")
        assert result.num_components == 1

    def test_both_returns_grayscale(self):
        img = _make_test_image()
        result = sobel(img, direction="both")
        assert result.num_components == 1

    def test_detects_edges(self):
        img = _make_edge_image()
        result = sobel(img, direction="both")
        # Edge coefficients should have non-zero AC content
        ac_energy = np.sum(result.y_coeffs[:, :, 1:, :].astype(np.float64) ** 2)
        assert ac_energy > 0

    def test_dimensions_preserved(self):
        img = _make_test_image()
        result = sobel(img)
        assert result.width == img.width
        assert result.height == img.height

    def test_invalid_direction_raises(self):
        img = _make_test_image()
        with pytest.raises(ValueError):
            sobel(img, direction="diagonal")

    def test_saveable(self, tmp_path):
        img = _make_test_image()
        result = sobel(img)
        out = tmp_path / "sobel.jpg"
        result.save(str(out))
        assert out.exists()


# -- Scharr --

class TestScharr:
    def test_returns_grayscale(self):
        img = _make_test_image()
        result = scharr(img, direction="horizontal")
        assert result.num_components == 1

    def test_detects_edges(self):
        img = _make_edge_image()
        result = scharr(img, direction="both")
        ac_energy = np.sum(result.y_coeffs[:, :, 1:, :].astype(np.float64) ** 2)
        assert ac_energy > 0

    def test_both_direction(self):
        img = _make_test_image()
        result = scharr(img, direction="both")
        assert result.num_components == 1


# -- Box blur --

class TestBoxBlur:
    def test_reduces_variance(self):
        img = _make_test_image()
        original_std = img.to_pixels().astype(float).std()
        result = box_blur(img, radius=2)
        result_std = result.to_pixels().astype(float).std()
        assert result_std < original_std

    def test_dc_preserved(self):
        img = _make_test_image()
        result = box_blur(img, radius=2)
        # Mean brightness should be roughly preserved
        orig_mean = img.to_pixels().astype(float).mean()
        result_mean = result.to_pixels().astype(float).mean()
        assert abs(orig_mean - result_mean) < 20

    def test_dimensions_preserved(self):
        img = _make_test_image()
        result = box_blur(img, radius=3)
        assert result.width == img.width
        assert result.height == img.height

    def test_invalid_radius_raises(self):
        img = _make_test_image()
        with pytest.raises(ValueError):
            box_blur(img, radius=0)


# -- Emboss --

class TestEmboss:
    def test_returns_grayscale(self):
        img = _make_test_image()
        result = emboss(img)
        assert result.num_components == 1

    def test_produces_relief_effect(self):
        img = _make_test_image()
        result = emboss(img)
        px = result.to_pixels()
        # Emboss should produce variation (not flat)
        assert px.std() > 1

    def test_different_angles(self):
        img = _make_test_image()
        r1 = emboss(img, angle=0)
        r2 = emboss(img, angle=90)
        # Different angles should produce different results
        assert not np.array_equal(r1.y_coeffs, r2.y_coeffs)


# -- Band-pass --

class TestBandpass:
    def test_removes_dc(self):
        img = _make_test_image()
        result = bandpass(img, low_cutoff=1, high_cutoff=6)
        # DC should be zeroed
        assert np.all(result.y_coeffs[:, :, 0, 0] == 0)

    def test_removes_high_freq(self):
        img = _make_test_image()
        result = bandpass(img, low_cutoff=0, high_cutoff=3)
        # High frequencies should be zeroed
        hf_energy = np.sum(result.y_coeffs[:, :, 5:, 5:].astype(np.float64) ** 2)
        assert hf_energy == 0

    def test_passthrough_full_range(self):
        img = _make_test_image()
        result = bandpass(img, low_cutoff=0, high_cutoff=7)
        # Full range should preserve everything
        np.testing.assert_array_equal(result.y_coeffs, img.y_coeffs)

    def test_invalid_cutoff_raises(self):
        img = _make_test_image()
        with pytest.raises(ValueError):
            bandpass(img, low_cutoff=5, high_cutoff=3)


# -- Unsharp mask --

class TestUnsharpMask:
    def test_increases_high_freq_energy(self):
        img = _make_test_image()
        result = unsharp_mask(img, sigma=2.0, amount=1.5)
        orig_hf = np.sum(img.y_coeffs[:, :, 3:, 3:].astype(np.float64) ** 2)
        result_hf = np.sum(result.y_coeffs[:, :, 3:, 3:].astype(np.float64) ** 2)
        assert result_hf > orig_hf

    def test_amount_0_is_identity(self):
        img = _make_test_image()
        result = unsharp_mask(img, sigma=2.0, amount=0.0)
        np.testing.assert_array_equal(result.y_coeffs, img.y_coeffs)

    def test_dc_preserved(self):
        img = _make_test_image()
        result = unsharp_mask(img, sigma=2.0, amount=1.5)
        np.testing.assert_array_equal(
            result.y_coeffs[:, :, 0, 0], img.y_coeffs[:, :, 0, 0]
        )

    def test_dimensions_preserved(self):
        img = _make_test_image()
        result = unsharp_mask(img, sigma=2.0, amount=1.5)
        assert result.width == img.width
        assert result.height == img.height
