"""Augmentation pipeline for DCTDataset.

Supports string and dict config formats with probability control.
All augmentations operate in DCT domain.

Reproducibility
---------------
The pipeline owns a per-process ``numpy.random.Generator``. With a fixed
``seed`` the augmentation stream is deterministic in a single process. Under
``DataLoader(num_workers>0)`` each worker mixes its worker id into the seed via
a ``SeedSequence``, so workers produce distinct (but individually reproducible)
streams instead of the identical or entropy-correlated draws you get from the
global ``numpy.random`` state after a fork.
"""

from __future__ import annotations

import os

import numpy as np

from dct_vision.core.dct_image import DCTImage

_INT32_MAX = 2**31 - 1


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
    seed : int, optional
        Base seed for reproducible augmentation. If None, entropy from the OS
        is used (non-deterministic). Distinct DataLoader workers derive distinct
        streams from this base automatically.
    """

    def __init__(self, specs: list, seed: int | None = None):
        self.augmentations = [_parse_augmentation(s) for s in specs]
        self._seed = seed
        self._rng: np.random.Generator | None = None
        self._pid: int | None = None

    def reseed(self, seed: int | None) -> None:
        """Reset the base seed and force the generator to be rebuilt."""
        self._seed = seed
        self._rng = None
        self._pid = None

    def _rng_for_process(self) -> np.random.Generator:
        """Return this process's generator, rebuilding it after a fork.

        DataLoader workers are forked children; each gets a distinct stream by
        folding its worker id into the SeedSequence.
        """
        pid = os.getpid()
        if self._rng is not None and self._pid == pid:
            return self._rng

        worker_id = 0
        try:  # torch is optional at import time
            from torch.utils.data import get_worker_info

            info = get_worker_info()
            if info is not None:
                worker_id = info.id
        except Exception:  # noqa: BLE001 - torch absent or not in a worker
            pass

        if self._seed is None:
            seq = np.random.SeedSequence()  # OS entropy
        else:
            seq = np.random.SeedSequence([int(self._seed), int(worker_id)])

        self._rng = np.random.default_rng(seq)
        self._pid = pid
        return self._rng

    def __call__(self, img: DCTImage) -> DCTImage:
        rng = self._rng_for_process()

        for aug in self.augmentations:
            name = aug["name"]
            p = aug.get("p", 1.0)

            if rng.random() > p:
                continue

            # Deterministic per-augmentation sub-seed drawn from the pipeline RNG.
            sub_seed = int(rng.integers(0, _INT32_MAX))

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
                img = brightness_jitter(img, max_offset=max_offset, seed=sub_seed)
            elif name == "contrast_jitter":
                from dct_vision.augment.jitter import contrast_jitter
                max_factor = aug.get("max_factor", 0.3)
                img = contrast_jitter(img, max_factor=max_factor, seed=sub_seed)
            elif name == "noise":
                from dct_vision.augment.noise import gaussian_noise
                sigma = aug.get("sigma", 3.0)
                img = gaussian_noise(img, sigma=sigma, seed=sub_seed)
            elif name == "crop":
                from dct_vision.augment.crop import block_crop
                block_rows = aug.get("block_rows", 4)
                block_cols = aug.get("block_cols", 4)
                bh, bw = img.y_coeffs.shape[:2]
                max_r = max(0, bh - block_rows)
                max_c = max(0, bw - block_cols)
                r = int(rng.integers(0, max_r + 1))
                c = int(rng.integers(0, max_c + 1))
                img = block_crop(img, r, c, block_rows, block_cols)

        return img
