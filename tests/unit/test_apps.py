"""Tests for practical DCT-native applications (dedup, forensics, thumbnail)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from dct_vision.apps.dedup import find_duplicates
from dct_vision.apps.forensics import detect_double_compression
from dct_vision.apps.thumbnail import dc_thumbnail
from dct_vision.core.dct_image import DCTImage


def _textured(h=256, w=256, seed=0):
    """A textured image whose *structure* depends on the seed."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    fx = 5 + (seed % 7) * 2          # seed changes spatial frequencies...
    fy = 4 + (seed % 5) * 3
    px = (seed * 1.3) % (2 * np.pi)  # ...and phase
    base = (128 + 50 * np.sin(xx / fx + px) + 35 * np.cos(yy / fy) + rng.normal(0, 8, (h, w)))
    base = base.clip(0, 255).astype(np.uint8)
    return np.stack([base, base, base], axis=-1)


def _save_jpeg(arr, path, quality):
    Image.fromarray(arr).save(str(path), "JPEG", quality=quality)


class TestDuplicateDetection:
    def test_finds_exact_duplicates(self, tmp_path):
        arr = _textured(seed=1)
        _save_jpeg(arr, tmp_path / "a.jpg", 90)
        _save_jpeg(arr, tmp_path / "b.jpg", 90)          # identical
        _save_jpeg(_textured(seed=99), tmp_path / "c.jpg", 90)  # different

        groups = find_duplicates(str(tmp_path), max_distance=2)
        assert len(groups) == 1
        names = {p.split("/")[-1] for p in groups[0]}
        assert names == {"a.jpg", "b.jpg"}

    def test_near_duplicate_recompressed(self, tmp_path):
        arr = _textured(seed=2)
        _save_jpeg(arr, tmp_path / "orig.jpg", 95)
        _save_jpeg(arr, tmp_path / "recompressed.jpg", 60)  # same image, lower quality
        groups = find_duplicates(str(tmp_path), max_distance=5)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_no_duplicates_returns_empty(self, tmp_path):
        for i in range(3):
            _save_jpeg(_textured(seed=i * 100 + 7), tmp_path / f"{i}.jpg", 90)
        groups = find_duplicates(str(tmp_path), max_distance=0)
        assert groups == []


class TestThumbnail:
    def test_dc_thumbnail_shape_color(self):
        img = DCTImage.from_array(_textured(256, 256), quality=90)
        thumb = dc_thumbnail(img)
        assert thumb.shape == (32, 32, 3)  # 256/8
        assert thumb.dtype == np.uint8

    def test_dc_thumbnail_grayscale(self):
        gray = _textured(128, 128)[:, :, 0]
        img = DCTImage.from_array(gray, quality=90)
        thumb = dc_thumbnail(img)
        assert thumb.shape == (16, 16)

    def test_resize_to_size(self):
        img = DCTImage.from_array(_textured(256, 512), quality=90)
        thumb = dc_thumbnail(img, size=64)
        assert max(thumb.shape[:2]) == 64

    def test_approximates_downscaled_image(self):
        """DC thumbnail should approximate an 8x block-averaged image."""
        arr = _textured(256, 256)
        img = DCTImage.from_array(arr, quality=95)
        thumb = dc_thumbnail(img)
        # Ground truth: 8x8 block mean of the luma channel.
        luma = arr[:, :, 0].astype(np.float64)
        block_mean = luma.reshape(32, 8, 32, 8).mean(axis=(1, 3))
        thumb_luma = thumb.mean(axis=-1).astype(np.float64)
        # Correlate strongly with the true block means.
        corr = np.corrcoef(block_mean.ravel(), thumb_luma.ravel())[0, 1]
        assert corr > 0.95


class TestDoubleCompression:
    def _load(self, arr, tmp_path, name, q1, q2=None):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, "JPEG", quality=q1)
        buf.seek(0)
        if q2 is not None:
            arr = np.array(Image.open(buf))
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, "JPEG", quality=q2)
            buf.seek(0)
        path = tmp_path / name
        path.write_bytes(buf.read())
        return DCTImage.from_file(str(path))

    def test_double_scores_higher_than_single(self, tmp_path):
        arr = _textured(seed=5)
        single = self._load(arr, tmp_path, "single.jpg", 90)
        double = self._load(arr, tmp_path, "double.jpg", 60, 90)  # coarse then fine
        s = detect_double_compression(single)["score"]
        d = detect_double_compression(double)["score"]
        assert d > s + 3.0

    def test_flags_double_compressed(self, tmp_path):
        arr = _textured(seed=6)
        double = self._load(arr, tmp_path, "double.jpg", 60, 92)
        result = detect_double_compression(double)
        assert result["is_double_compressed"] is True

    def test_returns_per_position(self, tmp_path):
        arr = _textured(seed=7)
        single = self._load(arr, tmp_path, "single.jpg", 85)
        result = detect_double_compression(single)
        assert "per_position" in result
        assert len(result["per_position"]) == 5
