"""PyTorch Dataset that yields DCT coefficient tensors.

Loads JPEG images via native libjpeg extraction and returns DCT
coefficients as PyTorch tensors -- no pixel decode in the data pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from dct_vision.core.dct_image import DCTImage
from dct_vision.utils.constants import BLOCK_SIZE

VALID_MODES = {"y_only", "ycbcr", "dc_only"}


def _scan_image_folder(root: str) -> tuple[list[str], list[int], list[str], dict[str, int]]:
    """Scan an ImageFolder-style directory.

    Returns (paths, labels, class_names, class_to_idx).
    If no subdirectories with images exist, treats all images as class 0.
    """
    root_path = Path(root)
    subdirs = sorted([
        d.name for d in root_path.iterdir()
        if d.is_dir() and any(f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp") for f in d.iterdir())
    ])

    if subdirs:
        class_to_idx = {name: idx for idx, name in enumerate(subdirs)}
        paths = []
        labels = []
        for cls_name in subdirs:
            cls_dir = root_path / cls_name
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    paths.append(str(f))
                    labels.append(class_to_idx[cls_name])
        return paths, labels, subdirs, class_to_idx
    else:
        # No class subdirs -- all images in root
        paths = sorted([
            str(f) for f in root_path.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ])
        return paths, [0] * len(paths), [], {"": 0}


def _load_dct(path: str) -> DCTImage:
    """Load image as DCTImage (JPEG native or convert)."""
    if path.lower().endswith((".jpg", ".jpeg")):
        return DCTImage.from_file(path)
    else:
        from dct_vision.io.convert import convert_to_dct
        return convert_to_dct(path)


def _coeffs_to_tensor_y_only(
    img: DCTImage,
    resize_blocks: tuple[int, int] | None,
) -> torch.Tensor:
    """Convert Y channel coefficients to (64, bh, bw) tensor."""
    y = img.y_coeffs.astype(np.float32)  # (bh, bw, 8, 8)
    bh, bw = y.shape[:2]

    # Reshape to (bh, bw, 64) then transpose to (64, bh, bw)
    flat = y.reshape(bh, bw, 64).transpose(2, 0, 1)  # (64, bh, bw)

    if resize_blocks is not None:
        flat = _resize_block_grid(flat, resize_blocks)

    return torch.from_numpy(flat.copy())


def _coeffs_to_tensor_ycbcr(
    img: DCTImage,
    resize_blocks: tuple[int, int] | None,
) -> torch.Tensor:
    """Convert Y+Cb+Cr coefficients to (192, bh, bw) tensor."""
    y = img.y_coeffs.astype(np.float32)
    bh, bw = y.shape[:2]
    y_flat = y.reshape(bh, bw, 64).transpose(2, 0, 1)

    if img.cb_coeffs is not None:
        cb = img.cb_coeffs.astype(np.float32)
        cr = img.cr_coeffs.astype(np.float32)

        # Upsample chroma to match luma block dimensions
        cb_bh, cb_bw = cb.shape[:2]
        if cb_bh != bh or cb_bw != bw:
            cb = np.repeat(np.repeat(cb, bh // cb_bh, axis=0), bw // cb_bw, axis=1)
            cr = np.repeat(np.repeat(cr, bh // cb_bh, axis=0), bw // cb_bw, axis=1)
            cb = cb[:bh, :bw]
            cr = cr[:bh, :bw]

        cb_flat = cb.reshape(bh, bw, 64).transpose(2, 0, 1)
        cr_flat = cr.reshape(bh, bw, 64).transpose(2, 0, 1)
        combined = np.concatenate([y_flat, cb_flat, cr_flat], axis=0)  # (192, bh, bw)
    else:
        # Grayscale -- pad with zeros for Cb/Cr
        zeros = np.zeros_like(y_flat)
        combined = np.concatenate([y_flat, zeros, zeros], axis=0)

    if resize_blocks is not None:
        combined = _resize_block_grid(combined, resize_blocks)

    return torch.from_numpy(combined.copy())


def _coeffs_to_tensor_dc_only(
    img: DCTImage,
    resize_blocks: tuple[int, int] | None,
) -> torch.Tensor:
    """Convert DC coefficients to (C, bh, bw) tensor. C=1 or C=3."""
    y_dc = img.y_coeffs[:, :, 0, 0].astype(np.float32)  # (bh, bw)

    if img.cb_coeffs is not None:
        bh, bw = y_dc.shape
        cb_dc = img.cb_coeffs[:, :, 0, 0].astype(np.float32)
        cr_dc = img.cr_coeffs[:, :, 0, 0].astype(np.float32)

        # Upsample chroma DC
        cb_bh, cb_bw = cb_dc.shape
        if cb_bh != bh or cb_bw != bw:
            cb_dc = np.repeat(np.repeat(cb_dc, bh // cb_bh, axis=0), bw // cb_bw, axis=1)
            cr_dc = np.repeat(np.repeat(cr_dc, bh // cb_bh, axis=0), bw // cb_bw, axis=1)
            cb_dc = cb_dc[:bh, :bw]
            cr_dc = cr_dc[:bh, :bw]

        combined = np.stack([y_dc, cb_dc, cr_dc], axis=0)  # (3, bh, bw)
    else:
        combined = y_dc[np.newaxis, :, :]  # (1, bh, bw)

    if resize_blocks is not None:
        combined = _resize_block_grid(combined, resize_blocks)

    return torch.from_numpy(combined.copy())


def _resize_block_grid(
    tensor: np.ndarray,
    target: tuple[int, int],
) -> np.ndarray:
    """Resize block grid (C, bh, bw) to (C, th, tw) via truncation or padding."""
    c, bh, bw = tensor.shape
    th, tw = target

    result = np.zeros((c, th, tw), dtype=tensor.dtype)
    copy_h = min(bh, th)
    copy_w = min(bw, tw)
    result[:, :copy_h, :copy_w] = tensor[:, :copy_h, :copy_w]

    return result


class DCTDataset(Dataset):
    """PyTorch Dataset yielding DCT coefficient tensors.

    Loads JPEG images and returns DCT coefficients as tensors instead
    of decoded pixel arrays. Supports ImageFolder-style class directories.

    Parameters
    ----------
    root : str
        Root directory. Either ImageFolder-style (root/class/img.jpg)
        or flat (root/img.jpg).
    mode : str
        Tensor format: 'y_only' (64,bh,bw), 'ycbcr' (192,bh,bw),
        'dc_only' (3,bh,bw).
    resize_blocks : tuple[int, int], optional
        Fixed output block grid size (bh, bw). Pads/truncates as needed.
    augmentations : list, optional
        Augmentation pipeline (see AugmentationPipeline).
    """

    def __init__(
        self,
        root: str,
        mode: str = "y_only",
        resize_blocks: tuple[int, int] | None = None,
        augmentations: list | None = None,
        seed: int | None = None,
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got '{mode}'")

        self.root = root
        self.mode = mode
        self.resize_blocks = resize_blocks

        self.paths, self.labels, self.classes, self.class_to_idx = _scan_image_folder(root)

        self._augmentations = None
        if augmentations:
            from dct_vision.ml.augment_pipeline import AugmentationPipeline
            self._augmentations = AugmentationPipeline(augmentations, seed=seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path = self.paths[idx]
        label = self.labels[idx]

        img = _load_dct(path)

        if self._augmentations is not None:
            img = self._augmentations(img)

        if self.mode == "y_only":
            tensor = _coeffs_to_tensor_y_only(img, self.resize_blocks)
        elif self.mode == "ycbcr":
            tensor = _coeffs_to_tensor_ycbcr(img, self.resize_blocks)
        elif self.mode == "dc_only":
            tensor = _coeffs_to_tensor_dc_only(img, self.resize_blocks)

        return tensor, label
