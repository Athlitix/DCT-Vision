"""Tests for cross-block boundary strategy."""

import numpy as np
import pytest
from PIL import Image, ImageFilter

from dct_vision.core.dct_image import DCTImage
from dct_vision.ops.cross_block import cross_block_blur, cross_block_edges


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)


def _spatial_blur(pixels, sigma):
    """Ground truth: Pillow Gaussian blur on pixels."""
    return np.array(
        Image.fromarray(pixels).filter(ImageFilter.GaussianBlur(radius=sigma)),
        dtype=np.uint8,
    )


class TestCrossBlockBlur:
    def test_returns_valid_image(self, sample_jpeg):
        img = DCTImage.from_file(str(sample_jpeg))
        result = cross_block_blur(img, sigma=3.0)
        assert result.width == img.width
        assert result.height == img.height
        assert result.y_coeffs.shape == img.y_coeffs.shape

    def test_cross_block_better_than_naive_at_high_sigma(self):
        """Cross-block blur should be closer to ground truth than naive blur
        when sigma is large enough to span block boundaries."""
        pixels = np.random.RandomState(42).randint(0, 256, (128, 128, 3), dtype=np.uint8)
        img = DCTImage.from_array(pixels, quality=95)

        # Ground truth
        ground_truth = _spatial_blur(pixels, sigma=4.0)

        # Naive DCT blur (no cross-block)
        from dct_vision.ops.blur import blur
        naive = blur(img, sigma=4.0)
        naive_pixels = naive.to_pixels()

        # Cross-block blur
        cb = cross_block_blur(img, sigma=4.0)
        cb_pixels = cb.to_pixels()

        # Trim to common size
        h = min(ground_truth.shape[0], naive_pixels.shape[0], cb_pixels.shape[0])
        w = min(ground_truth.shape[1], naive_pixels.shape[1], cb_pixels.shape[1])
        gt = ground_truth[:h, :w]

        naive_psnr = _psnr(gt, naive_pixels[:h, :w])
        cb_psnr = _psnr(gt, cb_pixels[:h, :w])

        # Cross-block should match ground truth better
        assert cb_psnr > naive_psnr, (
            f"Cross-block PSNR ({cb_psnr:.1f}dB) should beat "
            f"naive PSNR ({naive_psnr:.1f}dB)"
        )

    def test_small_sigma_similar_to_naive(self):
        """For small sigma (within one block), cross-block adds little benefit."""
        pixels = np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img = DCTImage.from_array(pixels, quality=95)

        from dct_vision.ops.blur import blur
        naive = blur(img, sigma=0.5)
        cb = cross_block_blur(img, sigma=0.5)

        # Both should produce similar results for small sigma
        naive_px = naive.to_pixels()
        cb_px = cb.to_pixels()
        h = min(naive_px.shape[0], cb_px.shape[0])
        w = min(naive_px.shape[1], cb_px.shape[1])
        psnr = _psnr(naive_px[:h, :w], cb_px[:h, :w])
        assert psnr > 10, f"Small sigma results diverged too much, PSNR={psnr:.1f}dB"

    @pytest.mark.parametrize("sigma", [1.0, 2.0, 4.0, 8.0])
    def test_psnr_vs_ground_truth_across_sigmas(self, sigma):
        """Cross-block blur should have reasonable PSNR vs spatial ground truth."""
        pixels = np.random.RandomState(42).randint(0, 256, (128, 128, 3), dtype=np.uint8)
        img = DCTImage.from_array(pixels, quality=95)
        ground_truth = _spatial_blur(pixels, sigma=sigma)
        cb = cross_block_blur(img, sigma=sigma)
        cb_pixels = cb.to_pixels()

        h = min(ground_truth.shape[0], cb_pixels.shape[0])
        w = min(ground_truth.shape[1], cb_pixels.shape[1])
        p = _psnr(ground_truth[:h, :w], cb_pixels[:h, :w])
        # Should be at least reasonable quality
        assert p > 15, f"PSNR too low at sigma={sigma}: {p:.1f}dB"

    def test_saveable(self, sample_jpeg, tmp_path):
        img = DCTImage.from_file(str(sample_jpeg))
        result = cross_block_blur(img, sigma=3.0)
        out = tmp_path / "cb_blur.jpg"
        result.save(str(out))
        assert out.exists()

    def test_grayscale(self, grayscale_jpeg):
        img = DCTImage.from_file(str(grayscale_jpeg))
        result = cross_block_blur(img, sigma=3.0)
        assert result.num_components == 1


class TestCrossBlockEdges:
    def test_returns_grayscale(self):
        pixels = np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img = DCTImage.from_array(pixels, quality=95)
        result = cross_block_edges(img)
        assert result.num_components == 1

    def test_detects_edges_across_boundaries(self):
        """An edge placed exactly on a block boundary should be detected."""
        # Create image with a sharp vertical edge at x=32 (block boundary)
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        pixels[:, 32:, :] = 255
        img = DCTImage.from_array(pixels, quality=95)

        result = cross_block_edges(img)
        edge_pixels = result.to_pixels()
        # The edge region (columns near 32) should have high values
        # relative to flat regions
        edge_col_energy = edge_pixels[:, 28:36].astype(float).std()
        flat_energy = edge_pixels[:, :16].astype(float).std()
        assert edge_col_energy > flat_energy
