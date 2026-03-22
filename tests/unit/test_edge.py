"""Tests for edge detection in DCT domain."""

import numpy as np
import pytest

from dct_vision.core.dct_image import DCTImage
from dct_vision.ops.edge import detect_edges


class TestDetectEdges:
    def test_laplacian_method(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = detect_edges(img, method="laplacian")
        assert result.y_coeffs is not None
        assert result.num_components == 1  # Edge map is grayscale

    def test_gradient_method(self):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = detect_edges(img, method="gradient")
        assert result.y_coeffs is not None
        assert result.num_components == 1

    def test_dc_suppressed(self):
        """Edge detection should suppress DC — flat image has no edges."""
        img = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        result = detect_edges(img, method="laplacian")
        # Flat image: edge coefficients should be ~0 (DC is zeroed by Laplacian)
        # to_pixels adds +128 level shift, so flat edges = 128
        pixels = result.to_pixels()
        assert pixels.std() < 5  # Uniform = no edges

    def test_edges_visible_on_textured_image(self):
        """Textured image should produce visible edges."""
        # Create image with intra-block edges (not aligned to 8px grid)
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        pixels[:, 32:, :] = 255  # Vertical edge at column 32
        pixels[20:44, :, :] = 200  # Horizontal band
        img = DCTImage.from_array(pixels, quality=95)
        result = detect_edges(img, method="laplacian")
        # Edge coefficients should have non-zero AC content
        ac_energy = np.sum(result.y_coeffs[:, :, 1:, :].astype(np.float64) ** 2)
        assert ac_energy > 0

    def test_returns_new_image(self):
        img = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        result = detect_edges(img, method="laplacian")
        assert result is not img

    def test_invalid_method_raises(self):
        img = DCTImage.from_array(
            np.full((64, 64, 3), 128, dtype=np.uint8), quality=95
        )
        with pytest.raises(ValueError):
            detect_edges(img, method="invalid")

    def test_saveable(self, tmp_path):
        img = DCTImage.from_array(
            np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8),
            quality=85,
        )
        result = detect_edges(img, method="laplacian")
        out = tmp_path / "edges.jpg"
        result.save(str(out))
        assert out.exists()
