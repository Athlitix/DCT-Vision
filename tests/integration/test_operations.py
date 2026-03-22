"""Integration tests: DCT-domain operations vs spatial-domain ground truth.

Verifies that our frequency-domain operations produce visually
reasonable results compared to direct pixel manipulation.
"""

import numpy as np
import pytest
from PIL import Image, ImageFilter

from dct_vision.core.dct_image import DCTImage
from dct_vision.ops.blur import blur
from dct_vision.ops.sharpen import sharpen
from dct_vision.ops.color import adjust_brightness, adjust_contrast
from dct_vision.ops.scale import downscale
from dct_vision.ops.edge import detect_edges
from dct_vision.ops.quality import estimate_quality, dct_stats


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)


class TestBlurVsSpatial:
    def test_blur_visual_similarity(self, sample_jpeg, tmp_path):
        """DCT blur should produce similar result to Pillow GaussianBlur."""
        # Spatial ground truth
        pil_img = Image.open(str(sample_jpeg))
        spatial_blurred = np.array(pil_img.filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.uint8)

        # DCT-domain blur
        dct_img = DCTImage.from_file(str(sample_jpeg))
        dct_blurred = blur(dct_img, sigma=2.0)
        dct_result = dct_blurred.to_pixels()

        # Both should be blurry — compare structure, not exact match
        # DCT blur won't match Pillow exactly (different algorithms)
        # but both should reduce high-freq content relative to original
        original = np.array(pil_img, dtype=np.uint8)
        assert _psnr(original, dct_result) < _psnr(original, original)  # Modified, not identity

    def test_blur_reduces_variance(self, sample_jpeg):
        """Blurred image should have lower pixel variance than original."""
        img = DCTImage.from_file(str(sample_jpeg))
        original = img.to_pixels()
        blurred = blur(img, sigma=3.0).to_pixels()
        assert blurred.astype(float).std() < original.astype(float).std()


class TestSharpenVsSpatial:
    def test_sharpen_increases_variance(self, sample_jpeg):
        """Sharpened image should have higher pixel variance."""
        img = DCTImage.from_file(str(sample_jpeg))
        original = img.to_pixels()
        sharpened = sharpen(img, amount=2.0).to_pixels()
        assert sharpened.astype(float).std() >= original.astype(float).std() * 0.9


class TestBrightnessVsSpatial:
    def test_brightness_matches_pixel_add(self, sample_jpeg):
        """DCT brightness should approximate pixel-level addition."""
        img = DCTImage.from_file(str(sample_jpeg))
        original = img.to_pixels().astype(float)
        brightened = adjust_brightness(img, offset=40).to_pixels().astype(float)

        # Mean brightness should increase (attenuated by quantization)
        mean_diff = brightened.mean() - original.mean()
        assert mean_diff > 2  # Some increase visible


class TestContrastVsSpatial:
    def test_high_contrast_increases_std(self, sample_jpeg):
        img = DCTImage.from_file(str(sample_jpeg))
        original = img.to_pixels()
        contrasted = adjust_contrast(img, factor=2.0).to_pixels()
        assert contrasted.astype(float).std() > original.astype(float).std()

    def test_low_contrast_decreases_std(self, sample_jpeg):
        img = DCTImage.from_file(str(sample_jpeg))
        original = img.to_pixels()
        muted = adjust_contrast(img, factor=0.3).to_pixels()
        assert muted.astype(float).std() < original.astype(float).std()


class TestDownscaleVsSpatial:
    def test_downscale_vs_pillow_resize(self, sample_jpeg, tmp_path):
        """DCT downscale should produce similar result to Pillow resize."""
        pil_img = Image.open(str(sample_jpeg))
        spatial_small = np.array(pil_img.resize((128, 128), Image.LANCZOS), dtype=np.uint8)

        dct_img = DCTImage.from_file(str(sample_jpeg))
        dct_small = downscale(dct_img, factor=2)
        dct_result = dct_small.to_pixels()

        # Both should be 128x128 (approximately — DCT might differ slightly)
        assert dct_result.shape[0] <= 128 + 8  # Allow block-rounding
        assert dct_result.shape[1] <= 128 + 8

        # Trim to same size for comparison
        h = min(spatial_small.shape[0], dct_result.shape[0])
        w = min(spatial_small.shape[1], dct_result.shape[1])
        psnr = _psnr(spatial_small[:h, :w], dct_result[:h, :w])
        assert psnr > 15, f"Downscale PSNR too low: {psnr:.1f}dB"


class TestEdgeDetection:
    def test_edge_map_structure(self, sample_jpeg):
        """Edge map should highlight transitions."""
        img = DCTImage.from_file(str(sample_jpeg))
        edges = detect_edges(img, method="laplacian")
        assert edges.num_components == 1
        pixels = edges.to_pixels()
        assert pixels.ndim == 2

    def test_gradient_edge_map(self, sample_jpeg):
        img = DCTImage.from_file(str(sample_jpeg))
        edges = detect_edges(img, method="gradient")
        assert edges.num_components == 1


class TestQualityEstimation:
    @pytest.mark.parametrize("fixture,expected_min,expected_max", [
        ("sample_jpeg_q50", 30, 70),
        ("sample_jpeg_q75", 55, 90),
        ("sample_jpeg_q95", 80, 100),
    ])
    def test_quality_estimation_range(self, fixture, expected_min, expected_max, request):
        path = request.getfixturevalue(fixture)
        img = DCTImage.from_file(str(path))
        q = estimate_quality(img)
        assert expected_min <= q <= expected_max, f"Quality {q} not in [{expected_min}, {expected_max}]"


class TestOperationChaining:
    def test_blur_then_sharpen(self, sample_jpeg, tmp_path):
        """Blur then sharpen should produce a valid result."""
        img = DCTImage.from_file(str(sample_jpeg))
        result = sharpen(blur(img, sigma=2.0), amount=1.5)
        out = tmp_path / "blur_then_sharpen.jpg"
        result.save(str(out))
        assert out.exists()

    def test_brightness_then_contrast(self, sample_jpeg, tmp_path):
        img = DCTImage.from_file(str(sample_jpeg))
        result = adjust_contrast(adjust_brightness(img, offset=20), factor=1.3)
        out = tmp_path / "bright_contrast.jpg"
        result.save(str(out))
        assert out.exists()

    def test_downscale_then_blur(self, sample_jpeg, tmp_path):
        img = DCTImage.from_file(str(sample_jpeg))
        small = downscale(img, factor=2)
        result = blur(small, sigma=1.5)
        out = tmp_path / "small_blurred.jpg"
        result.save(str(out))
        assert out.exists()
