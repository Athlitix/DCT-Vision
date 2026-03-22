"""Tests for PyTorch DCTDataset."""

import numpy as np
import pytest
import shutil
from pathlib import Path
from PIL import Image

torch = pytest.importorskip("torch")

from dct_vision.ml.dataset import DCTDataset


@pytest.fixture
def image_folder(tmp_path):
    """Create an ImageFolder-style directory with 2 classes."""
    for cls_name in ["cat", "dog"]:
        cls_dir = tmp_path / cls_name
        cls_dir.mkdir()
        rng = np.random.RandomState(hash(cls_name) % 2**31)
        for i in range(5):
            px = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(px).save(str(cls_dir / f"{i:03d}.jpg"), quality=85)
    return tmp_path


@pytest.fixture
def single_class_folder(tmp_path):
    """Folder with images, no class subdirs."""
    for i in range(3):
        px = np.random.RandomState(i).randint(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(px).save(str(tmp_path / f"img_{i}.jpg"), quality=85)
    return tmp_path


class TestDCTDatasetBasic:
    def test_len(self, image_folder):
        ds = DCTDataset(root=str(image_folder))
        assert len(ds) == 10  # 5 cat + 5 dog

    def test_getitem_returns_tuple(self, image_folder):
        ds = DCTDataset(root=str(image_folder))
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 2  # (tensor, label)

    def test_tensor_is_float32(self, image_folder):
        ds = DCTDataset(root=str(image_folder))
        tensor, label = ds[0]
        assert tensor.dtype == torch.float32

    def test_label_is_int(self, image_folder):
        ds = DCTDataset(root=str(image_folder))
        _, label = ds[0]
        assert isinstance(label, int)

    def test_classes_detected(self, image_folder):
        ds = DCTDataset(root=str(image_folder))
        assert len(ds.classes) == 2
        assert "cat" in ds.classes
        assert "dog" in ds.classes

    def test_class_to_idx(self, image_folder):
        ds = DCTDataset(root=str(image_folder))
        assert isinstance(ds.class_to_idx, dict)
        assert len(ds.class_to_idx) == 2


class TestDCTDatasetModes:
    def test_y_only_shape(self, image_folder):
        ds = DCTDataset(root=str(image_folder), mode="y_only")
        tensor, _ = ds[0]
        # y_only: (64, bh, bw) -- 64 coefficients as channels
        assert tensor.shape[0] == 64

    def test_ycbcr_shape(self, image_folder):
        ds = DCTDataset(root=str(image_folder), mode="ycbcr")
        tensor, _ = ds[0]
        # ycbcr: (192, bh, bw) -- 64*3 channels
        assert tensor.shape[0] == 192

    def test_dc_only_shape(self, image_folder):
        ds = DCTDataset(root=str(image_folder), mode="dc_only")
        tensor, _ = ds[0]
        # dc_only: (3, bh, bw) -- just DC values
        assert tensor.shape[0] == 3

    def test_invalid_mode_raises(self, image_folder):
        with pytest.raises(ValueError):
            DCTDataset(root=str(image_folder), mode="invalid")


class TestDCTDatasetResize:
    def test_resize_blocks_fixed_output(self, image_folder):
        ds = DCTDataset(root=str(image_folder), mode="y_only", resize_blocks=(4, 4))
        tensor, _ = ds[0]
        assert tensor.shape == (64, 4, 4)

    def test_resize_blocks_consistent(self, image_folder):
        ds = DCTDataset(root=str(image_folder), mode="y_only", resize_blocks=(4, 4))
        shapes = set()
        for i in range(len(ds)):
            t, _ = ds[i]
            shapes.add(t.shape)
        assert len(shapes) == 1  # all same shape


class TestDCTDatasetDataLoader:
    def test_works_with_dataloader(self, image_folder):
        from torch.utils.data import DataLoader
        ds = DCTDataset(root=str(image_folder), mode="y_only", resize_blocks=(4, 4))
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch_tensor, batch_labels = next(iter(loader))
        assert batch_tensor.shape[0] == 4
        assert batch_labels.shape[0] == 4

    def test_multiple_workers(self, image_folder):
        from torch.utils.data import DataLoader
        ds = DCTDataset(root=str(image_folder), mode="y_only", resize_blocks=(4, 4))
        loader = DataLoader(ds, batch_size=2, num_workers=0)
        count = 0
        for batch in loader:
            count += batch[0].shape[0]
        assert count == 10


class TestDCTDatasetNoLabels:
    def test_single_folder_no_classes(self, single_class_folder):
        ds = DCTDataset(root=str(single_class_folder))
        assert len(ds) == 3
        _, label = ds[0]
        assert label == 0  # default label when no class structure
