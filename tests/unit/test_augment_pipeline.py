"""Tests for augmentation pipeline."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dct_vision.core.dct_image import DCTImage
from dct_vision.ml.augment_pipeline import AugmentationPipeline
from dct_vision.ml.dataset import DCTDataset
from PIL import Image


def _make_img():
    px = np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8)
    return DCTImage.from_array(px, quality=95)


class TestAugmentationPipeline:
    def test_string_config(self):
        pipe = AugmentationPipeline(["hflip"])
        img = _make_img()
        result = pipe(img)
        assert result.width == img.width

    def test_dict_config(self):
        pipe = AugmentationPipeline([{"name": "hflip", "p": 1.0}])
        img = _make_img()
        result = pipe(img)
        assert result.width == img.width

    def test_string_with_params(self):
        pipe = AugmentationPipeline(["noise:sigma=3.0"])
        img = _make_img()
        result = pipe(img)
        assert result.width == img.width

    def test_probability_0_skips(self):
        pipe = AugmentationPipeline(["hflip:p=0"])
        img = _make_img()
        np.random.seed(42)
        result = pipe(img)
        np.testing.assert_array_equal(result.y_coeffs, img.y_coeffs)

    def test_chaining(self):
        pipe = AugmentationPipeline(["hflip", "noise:sigma=2.0"])
        img = _make_img()
        result = pipe(img)
        assert result.width == img.width

    def test_crop_augmentation(self):
        pipe = AugmentationPipeline(["crop:block_rows=4,block_cols=4"])
        img = _make_img()
        np.random.seed(42)
        result = pipe(img)
        assert result.y_coeffs.shape[0] == 4
        assert result.y_coeffs.shape[1] == 4


class TestDatasetWithAugmentations:
    def test_augmented_dataset(self, tmp_path):
        # Create test images
        cls_dir = tmp_path / "cls"
        cls_dir.mkdir()
        for i in range(3):
            px = np.random.RandomState(i).randint(0, 256, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(px).save(str(cls_dir / f"{i}.jpg"), quality=85)

        ds = DCTDataset(
            root=str(tmp_path),
            mode="y_only",
            resize_blocks=(4, 4),
            augmentations=["hflip:p=0.5", "noise:sigma=2.0"],
        )
        tensor, label = ds[0]
        assert tensor.shape == (64, 4, 4)
