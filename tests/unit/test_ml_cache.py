"""Tests for DCT dataset caching."""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from dct_vision.ml.cache import prepare_cache, load_cached, dataset_info


@pytest.fixture
def sample_dir(tmp_path):
    cls = tmp_path / "src" / "cat"
    cls.mkdir(parents=True)
    for i in range(3):
        px = np.random.RandomState(i).randint(0, 256, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(px).save(str(cls / f"{i}.jpg"), quality=85)
    return tmp_path / "src"


class TestPrepareCache:
    def test_creates_npz_files(self, sample_dir, tmp_path):
        dst = tmp_path / "cache"
        stats = prepare_cache(str(sample_dir), str(dst))
        assert stats["count"] == 3
        npz_files = list(Path(dst).rglob("*.npz"))
        assert len(npz_files) == 3

    def test_npz_loadable(self, sample_dir, tmp_path):
        dst = tmp_path / "cache"
        prepare_cache(str(sample_dir), str(dst))
        npz_files = list(Path(dst).rglob("*.npz"))
        img = load_cached(str(npz_files[0]))
        assert img.y_coeffs is not None
        assert img.width == 32
        assert img.height == 32

    def test_roundtrip_coefficients(self, sample_dir, tmp_path):
        from dct_vision.core.dct_image import DCTImage
        # Use a specific known file
        src_file = sample_dir / "cat" / "0.jpg"
        src_img = DCTImage.from_file(str(src_file))

        dst = tmp_path / "cache"
        prepare_cache(str(sample_dir), str(dst))
        cached_file = dst / "cat" / "0.npz"
        cached_img = load_cached(str(cached_file))

        np.testing.assert_array_equal(src_img.y_coeffs, cached_img.y_coeffs)


class TestDatasetInfo:
    def test_returns_dict(self, sample_dir):
        info = dataset_info(str(sample_dir))
        assert "total_images" in info
        assert info["total_images"] == 3

    def test_detects_classes(self, sample_dir):
        info = dataset_info(str(sample_dir))
        assert info["classes"] == 1
        assert "cat" in info["class_names"]
