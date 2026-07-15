"""Tests for lossless DCT-domain rotation and transpose.

These are exact coefficient permutations (jpegtran-style), so decoding the
transformed coefficients must match applying the same geometric transform to
the decoded pixels -- up to identical clip/round -- i.e. very high PSNR.
"""

from __future__ import annotations

import numpy as np
import pytest

from dct_vision.core.dct_image import DCTImage
from dct_vision.math.metrics import psnr
from dct_vision.ops.geometry import rotate, rotate90, rotate180, rotate270, transpose


def _asymmetric_gray(h=64, w=64):
    """A non-symmetric pattern so orientation errors are detectable."""
    yy, xx = np.mgrid[0:h, 0:w]
    img = (xx * 2 + yy * 3) % 256
    img[: h // 4, : w // 4] = 250  # bright corner marker (top-left)
    return img.astype(np.uint8)


@pytest.fixture
def gray_img():
    return DCTImage.from_array(_asymmetric_gray(64, 64), quality=100)


@pytest.fixture
def rect_img():
    return DCTImage.from_array(_asymmetric_gray(48, 80), quality=100)


class TestTranspose:
    def test_matches_pixel_transpose(self, gray_img):
        px = gray_img.to_pixels()
        out = transpose(gray_img).to_pixels()
        assert psnr(px.T, out) > 55

    def test_swaps_dimensions(self, rect_img):
        out = transpose(rect_img)
        assert out.width == rect_img.height
        assert out.height == rect_img.width

    def test_rectangular_matches_pixels(self, rect_img):
        px = rect_img.to_pixels()
        out = transpose(rect_img).to_pixels()
        assert psnr(px.T, out) > 55

    def test_double_transpose_is_identity(self, gray_img):
        px = gray_img.to_pixels()
        out = transpose(transpose(gray_img)).to_pixels()
        assert psnr(px, out) > 55


class TestRotate180:
    def test_matches_pixel_rot180(self, gray_img):
        px = gray_img.to_pixels()
        out = rotate180(gray_img).to_pixels()
        assert psnr(np.rot90(px, 2), out) > 55

    def test_rectangular(self, rect_img):
        px = rect_img.to_pixels()
        rotated = rotate180(rect_img)
        assert rotated.width == rect_img.width and rotated.height == rect_img.height
        assert psnr(np.rot90(px, 2), rotated.to_pixels()) > 55


class TestRotate90:
    def test_cw_matches_numpy(self, gray_img):
        px = gray_img.to_pixels()
        out = rotate90(gray_img).to_pixels()  # clockwise
        assert psnr(np.rot90(px, -1), out) > 55

    def test_ccw_matches_numpy(self, gray_img):
        px = gray_img.to_pixels()
        out = rotate270(gray_img).to_pixels()  # 270 CW == 90 CCW
        assert psnr(np.rot90(px, 1), out) > 55

    def test_four_rotations_identity(self, gray_img):
        px = gray_img.to_pixels()
        out = rotate90(rotate90(rotate90(rotate90(gray_img)))).to_pixels()
        assert psnr(px, out) > 55

    def test_rectangular_swaps_dims(self, rect_img):
        out = rotate90(rect_img)
        assert out.width == rect_img.height
        assert out.height == rect_img.width


class TestRotateDispatch:
    @pytest.mark.parametrize("deg,k", [(0, 0), (90, -1), (180, 2), (270, 1), (-90, 1)])
    def test_dispatch(self, gray_img, deg, k):
        px = gray_img.to_pixels()
        out = rotate(gray_img, deg).to_pixels()
        assert psnr(np.rot90(px, k), out) > 55

    def test_invalid_angle_raises(self, gray_img):
        with pytest.raises(ValueError):
            rotate(gray_img, 45)


class TestColorRotation:
    def test_color_rot90_matches_pixels(self):
        rng = np.random.default_rng(1)
        px = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        img = DCTImage.from_array(px, quality=100)
        decoded = img.to_pixels()
        out = rotate90(img).to_pixels()
        assert psnr(np.rot90(decoded, -1), out) > 45
