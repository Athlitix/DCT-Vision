"""Augmentation pipeline for DCTDataset.

Supports string and dict config formats with probability control.
All augmentations operate in DCT domain.
"""

from __future__ import annotations

import numpy as np

from dct_vision.core.dct_image import DCTImage


def _parse_augmentation(spec) -> dict:
    """Parse augmentation spec from string or dict format."""
    if isinstance(spec, dict):
        return spec

    # String format: "name:key=val,key2=val2" or just "name"
    parts = spec.split(":")
    name = parts[0]
    params = {}
    if len(parts) > 1:
        for kv in parts[1].split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    v = float(v)
                    if v == int(v):
                        v = int(v)
                except ValueError:
                    pass
                params[k] = v
    params["name"] = name
    return params


class AugmentationPipeline:
    """Chain of DCT-domain augmentations with probability control.

    Parameters
    ----------
    specs : list
        List of augmentation specs. Each can be:
        - str: "hflip", "hflip:p=0.5", "noise:sigma=3.0"
        - dict: {"name": "hflip", "p": 0.5}
    """

    def __init__(self, specs: list):
        self.augmentations = [_parse_augmentation(s) for s in specs]

    def __call__(self, img: DCTImage) -> DCTImage:
        for aug in self.augmentations:
            name = aug["name"]
            p = aug.get("p", 1.0)

            if np.random.random() > p:
                continue

            if name == "hflip":
                from dct_vision.augment.flip import horizontal_flip
                img = horizontal_flip(img)
            elif name == "vflip":
                from dct_vision.augment.flip import vertical_flip
                img = vertical_flip(img)
            elif name in ("rotate", "rot90", "rot180", "rot270"):
                from dct_vision.ops.geometry import rotate as rotate_op
                degrees = aug.get("degrees", {"rot90": 90, "rot180": 180, "rot270": 270}.get(name, 90))
                img = rotate_op(img, int(degrees))
            elif name == "brightness_jitter":
                from dct_vision.augment.jitter import brightness_jitter
                max_offset = aug.get("max_offset", 20)
                img = brightness_jitter(img, max_offset=max_offset)
            elif name == "contrast_jitter":
                from dct_vision.augment.jitter import contrast_jitter
                max_factor = aug.get("max_factor", 0.3)
                img = contrast_jitter(img, max_factor=max_factor)
            elif name == "noise":
                from dct_vision.augment.noise import gaussian_noise
                sigma = aug.get("sigma", 3.0)
                img = gaussian_noise(img, sigma=sigma)
            elif name == "crop":
                from dct_vision.augment.crop import block_crop
                block_rows = aug.get("block_rows", 4)
                block_cols = aug.get("block_cols", 4)
                bh, bw = img.y_coeffs.shape[:2]
                max_r = max(0, bh - block_rows)
                max_c = max(0, bw - block_cols)
                r = np.random.randint(0, max_r + 1)
                c = np.random.randint(0, max_c + 1)
                img = block_crop(img, r, c, block_rows, block_cols)

        return img
