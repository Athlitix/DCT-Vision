"""DCT-domain augmentations vs their pixel-domain equivalents.

These tests prove mathematical equivalence (via PSNR against the pixel-domain
result), not merely that "something changed". Flips are exact coefficient
manipulations, so decoding a DCT-flipped image must match flipping the decoded
pixels. Noise is validated statistically (DC/mean preserved, energy added,
reproducible under a seed).
"""

from __future__ import annotations

import numpy as np
import pytest

from dct_vision.augment.flip import horizontal_flip, vertical_flip
from dct_vision.augment.noise import gaussian_noise
from dct_vision.core.dct_image import DCTImage
from dct_vision.math.metrics import psnr


def _pattern(h=64, w=64, color=False):
    yy, xx = np.mgrid[0:h, 0:w]
    base = ((xx * 3 + yy * 5) % 256).astype(np.uint8)
    base[: h // 5, : w // 5] = 240  # asymmetric marker
    if color:
        b = base.astype(np.int32)
        return np.stack([b, (b + 40) % 256, (b + 80) % 256], axis=-1).astype(np.uint8)
    return base


class TestFlipEquivalence:
    @pytest.mark.parametrize("color", [False, True])
    def test_hflip_matches_pixel_fliplr(self, color):
        img = DCTImage.from_array(_pattern(64, 64, color), quality=100)
        decoded = img.to_pixels()
        out = horizontal_flip(img).to_pixels()
        assert psnr(np.fliplr(decoded), out) > 55

    @pytest.mark.parametrize("color", [False, True])
    def test_vflip_matches_pixel_flipud(self, color):
        img = DCTImage.from_array(_pattern(64, 64, color), quality=100)
        decoded = img.to_pixels()
        out = vertical_flip(img).to_pixels()
        assert psnr(np.flipud(decoded), out) > 55

    def test_hflip_rectangular(self):
        img = DCTImage.from_array(_pattern(48, 80), quality=100)
        decoded = img.to_pixels()
        out = horizontal_flip(img).to_pixels()
        assert psnr(np.fliplr(decoded), out) > 55

    def test_hv_flip_equals_rot180(self):
        img = DCTImage.from_array(_pattern(64, 64), quality=100)
        decoded = img.to_pixels()
        out = vertical_flip(horizontal_flip(img)).to_pixels()
        assert psnr(np.rot90(decoded, 2), out) > 55


class TestTensorHFlipExact:
    """The tensor-level DCT hflip used in training must equal the real DCT flip."""

    @pytest.mark.parametrize("mode,channels", [("y_only", 64), ("ycbcr", 192)])
    def test_tensor_hflip_matches_coeff_flip(self, mode, channels):
        import torch

        from dct_vision.ml.dataset import _coeffs_to_tensor_y_only, _coeffs_to_tensor_ycbcr
        from dct_vision.ml.train import _dct_hflip_signs

        to_tensor = {"y_only": _coeffs_to_tensor_y_only, "ycbcr": _coeffs_to_tensor_ycbcr}[mode]
        img = DCTImage.from_array(_pattern(64, 64, color=True), quality=100)

        base = to_tensor(img, None)                       # (C, bh, bw)
        flipped_via_coeffs = to_tensor(horizontal_flip(img), None)
        flipped_via_tensor = base.flip(-1) * _dct_hflip_signs(channels)

        assert base.shape[0] == channels
        assert torch.allclose(flipped_via_tensor, flipped_via_coeffs, atol=1e-4)


class TestNoiseProperties:
    def test_dc_preserved(self):
        img = DCTImage.from_array(_pattern(64, 64), quality=90)
        out = gaussian_noise(img, sigma=5.0, seed=0)
        np.testing.assert_array_equal(out.y_coeffs[:, :, 0, 0], img.y_coeffs[:, :, 0, 0])

    def test_mean_brightness_preserved(self):
        img = DCTImage.from_array(np.full((64, 64), 120, dtype=np.uint8), quality=95)
        before = img.to_pixels().astype(np.float64).mean()
        after = gaussian_noise(img, sigma=6.0, seed=1).to_pixels().astype(np.float64).mean()
        # DC untouched -> block means preserved -> global mean within rounding.
        assert abs(after - before) < 2.0

    def test_adds_energy(self):
        img = DCTImage.from_array(_pattern(64, 64), quality=90)
        ac_before = np.sum(img.y_coeffs[:, :, 1:, :].astype(np.float64) ** 2)
        out = gaussian_noise(img, sigma=8.0, seed=2)
        ac_after = np.sum(out.y_coeffs[:, :, 1:, :].astype(np.float64) ** 2)
        assert ac_after > ac_before

    def test_seed_reproducible(self):
        img = DCTImage.from_array(_pattern(64, 64), quality=90)
        a = gaussian_noise(img, sigma=5.0, seed=42)
        b = gaussian_noise(img, sigma=5.0, seed=42)
        np.testing.assert_array_equal(a.y_coeffs, b.y_coeffs)

    def test_different_seeds_differ(self):
        img = DCTImage.from_array(_pattern(64, 64), quality=90)
        a = gaussian_noise(img, sigma=5.0, seed=1)
        b = gaussian_noise(img, sigma=5.0, seed=2)
        assert not np.array_equal(a.y_coeffs, b.y_coeffs)
