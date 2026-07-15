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


class TestReproducibility:
    def test_same_seed_identical(self):
        specs = ["hflip:p=0.5", "noise:sigma=3.0", "brightness_jitter:max_offset=20"]
        img = _make_img()
        a = AugmentationPipeline(specs, seed=123)(img)
        b = AugmentationPipeline(specs, seed=123)(img)
        np.testing.assert_array_equal(a.y_coeffs, b.y_coeffs)

    def test_different_seed_differs(self):
        specs = ["noise:sigma=5.0"]
        img = _make_img()
        a = AugmentationPipeline(specs, seed=1)(img)
        b = AugmentationPipeline(specs, seed=2)(img)
        assert not np.array_equal(a.y_coeffs, b.y_coeffs)

    def test_sequence_reproducible_across_calls(self):
        specs = ["noise:sigma=4.0"]
        img = _make_img()
        p1 = AugmentationPipeline(specs, seed=7)
        p2 = AugmentationPipeline(specs, seed=7)
        seq1 = [p1(img).y_coeffs.copy() for _ in range(3)]
        seq2 = [p2(img).y_coeffs.copy() for _ in range(3)]
        for a, b in zip(seq1, seq2):
            np.testing.assert_array_equal(a, b)

    def test_reseed_resets_stream(self):
        specs = ["noise:sigma=4.0"]
        img = _make_img()
        pipe = AugmentationPipeline(specs, seed=9)
        first = pipe(img).y_coeffs.copy()
        pipe.reseed(9)
        again = pipe(img).y_coeffs.copy()
        np.testing.assert_array_equal(first, again)

    def test_worker_ids_diverge(self):
        """Different DataLoader workers must not produce identical augmentations."""
        specs = ["noise:sigma=5.0"]
        img = _make_img()

        class _FakeInfo:
            def __init__(self, wid):
                self.id = wid

        import dct_vision.ml.augment_pipeline as ap

        outputs = []
        for wid in (0, 1):
            pipe = AugmentationPipeline(specs, seed=100)
            # Simulate being inside a specific DataLoader worker.
            import torch.utils.data as tud
            orig = tud.get_worker_info
            tud.get_worker_info = lambda w=wid: _FakeInfo(w)
            try:
                outputs.append(pipe(img).y_coeffs.copy())
            finally:
                tud.get_worker_info = orig
        assert not np.array_equal(outputs[0], outputs[1])


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
