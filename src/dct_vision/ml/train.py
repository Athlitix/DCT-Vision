"""Training / benchmarking harness for DCT vs pixel image classification.

Covers the ML-2 validation (does a DCT-input CNN train as well as the same
pixel CNN?) and an ML-3-style augmentation comparison (pixel hflip vs DCT-domain
hflip), reporting accuracy, wall-clock training time, and data-loading time.

DCT tensors are pre-computed once into memory (see PrecomputedDCTDataset) so the
data pipeline does no per-epoch DCT work -- this is the caching win from ML-1c.

Usage:
    uv run python -m dct_vision.ml.train --dataset cifar10 --epochs 10
    uv run python -m dct_vision.ml.train --models pixelresnet,dctresnet --augment \
        --dataset cifar10 --epochs 15 --output benchmarks/results/ml_train.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from dct_vision.ml.models import (
    PixelCNN,
    DCTVanillaCNN,
    DCTFreqBranchCNN,
    PixelResNet18,
    DCTResNet18,
)


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def _load_pixels(dataset: str, root: str):
    """Return (train_px_uint8, train_y, test_px_uint8, test_y, num_classes, img_size).

    Pixels are returned as uint8 NHWC arrays so both the pixel and DCT paths
    start from identical decoded images.
    """
    from torchvision import datasets

    if dataset == "cifar10":
        tr = datasets.CIFAR10(root, train=True, download=True)
        te = datasets.CIFAR10(root, train=False, download=True)
        tr_x = tr.data.astype(np.uint8)                       # (N,32,32,3)
        te_x = te.data.astype(np.uint8)
        tr_y = np.array(tr.targets, dtype=np.int64)
        te_y = np.array(te.targets, dtype=np.int64)
        return tr_x, tr_y, te_x, te_y, 10, 32

    if dataset == "stl10":
        tr = datasets.STL10(root, split="train", download=True)
        te = datasets.STL10(root, split="test", download=True)
        tr_x = tr.data.transpose(0, 2, 3, 1).astype(np.uint8)  # (N,96,96,3)
        te_x = te.data.transpose(0, 2, 3, 1).astype(np.uint8)
        return tr_x, tr.labels.astype(np.int64), te_x, te.labels.astype(np.int64), 10, 96

    raise ValueError(f"unknown dataset {dataset}")


def _precompute_dct_tensors(pixels_uint8: np.ndarray, block_grid, mode: str) -> torch.Tensor:
    """Convert a stack of uint8 NHWC images to DCT tensors once.

    Returns a float32 tensor (N, C, bh, bw). This is the ML-1c pre-computation:
    the expensive DCT extraction happens a single time, not every epoch.
    """
    from dct_vision.core.dct_image import DCTImage
    from dct_vision.ml.dataset import (
        _coeffs_to_tensor_y_only,
        _coeffs_to_tensor_ycbcr,
        _coeffs_to_tensor_dc_only,
    )

    fn = {
        "y_only": _coeffs_to_tensor_y_only,
        "ycbcr": _coeffs_to_tensor_ycbcr,
        "dc_only": _coeffs_to_tensor_dc_only,
    }[mode]

    n = pixels_uint8.shape[0]
    out = []
    t0 = time.perf_counter()
    for i in range(n):
        dct_img = DCTImage.from_array(pixels_uint8[i], quality=95)
        out.append(fn(dct_img, block_grid))
        if (i + 1) % 5000 == 0:
            print(f"    precomputed {i+1}/{n} ({time.perf_counter()-t0:.1f}s)")
    return torch.stack(out)


# DCT-domain hflip on a y_only tensor: channel c encodes coefficient (u=c//8,
# v=c%8). A horizontal flip reverses the width axis and negates channels whose
# horizontal frequency index v is odd. This is the exact DCT flip, on tensors.
def _y_only_hflip_signs() -> torch.Tensor:
    v = torch.arange(64) % 8
    return torch.where(v % 2 == 1, -1.0, 1.0).view(64, 1, 1)


class DCTHFlipDataset(Dataset):
    """Wrap a precomputed y_only DCT TensorDataset with random DCT-domain hflip."""

    def __init__(self, tensors: torch.Tensor, labels: torch.Tensor, p: float = 0.5, seed: int = 0):
        self.tensors = tensors
        self.labels = labels
        self.p = p
        self.signs = _y_only_hflip_signs()
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.tensors.shape[0]

    def __getitem__(self, idx):
        x = self.tensors[idx]
        if self.rng.random() < self.p:
            x = x.flip(-1) * self.signs
        return x, self.labels[idx]


class PixelAugDataset(Dataset):
    """Pixel dataset with random hflip + reflect-pad random crop (CIFAR style)."""

    def __init__(self, pixels_uint8: np.ndarray, labels: np.ndarray, img_size: int, seed: int = 0):
        from torchvision import transforms

        self.pixels = pixels_uint8
        self.labels = labels
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=4, padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return self.pixels.shape[0]

    def __getitem__(self, idx):
        return self.tf(self.pixels[idx]), int(self.labels[idx])


def _pixel_tensors(pixels_uint8: np.ndarray) -> torch.Tensor:
    """Normalized (N,3,H,W) float tensor, no augmentation."""
    x = torch.from_numpy(pixels_uint8).permute(0, 3, 1, 2).float() / 255.0
    return (x - 0.5) / 0.5


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    data_time = compute_time = 0.0

    t0 = time.perf_counter()
    for batch_x, batch_y in loader:
        data_time += time.perf_counter() - t0
        t1 = time.perf_counter()

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        correct += output.argmax(1).eq(batch_y).sum().item()
        total += batch_x.size(0)
        compute_time += time.perf_counter() - t1
        t0 = time.perf_counter()

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "data_time_ms": data_time * 1000,
        "compute_time_ms": compute_time * 1000,
    }


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            correct += model(batch_x).argmax(1).eq(batch_y).sum().item()
            total += batch_x.size(0)
    return correct / total


def run_experiment(name, model, train_loader, test_loader, epochs, device, lr=0.001):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    params = sum(p.numel() for p in model.parameters())
    print(f"\n--- {name} ({params:,} params) ---")

    total_data_ms = 0.0
    start = time.perf_counter()
    for epoch in range(epochs):
        stats = train_epoch(model, train_loader, optimizer, criterion, device)
        total_data_ms += stats["data_time_ms"]
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            acc = evaluate(model, test_loader, device)
            print(f"  epoch {epoch+1}/{epochs}: train_acc={stats['accuracy']:.3f} "
                  f"test_acc={acc:.3f} loss={stats['loss']:.3f} "
                  f"data={stats['data_time_ms']:.0f}ms compute={stats['compute_time_ms']:.0f}ms")
    total_time = time.perf_counter() - start
    final_acc = evaluate(model, test_loader, device)
    print(f"  final: test_acc={final_acc:.4f} time={total_time:.1f}s data={total_data_ms/1000:.1f}s")
    return {
        "name": name,
        "params": params,
        "final_test_accuracy": round(final_acc, 4),
        "total_time_sec": round(total_time, 1),
        "total_data_time_sec": round(total_data_ms / 1000, 1),
    }


def _make_model(key, num_classes, block_grid, img_size, dct_channels):
    if key == "pixelcnn":
        return PixelCNN(num_classes=num_classes, input_channels=3, input_size=img_size)
    if key == "dctcnn":
        return DCTVanillaCNN(num_classes=num_classes, input_channels=dct_channels, block_grid=block_grid)
    if key == "dctfreq":
        return DCTFreqBranchCNN(num_classes=num_classes, block_grid=block_grid)
    if key == "pixelresnet":
        return PixelResNet18(num_classes=num_classes, input_channels=3)
    if key == "dctresnet":
        return DCTResNet18(num_classes=num_classes, input_channels=dct_channels, block_grid=block_grid)
    raise ValueError(f"unknown model {key}")


def main():
    p = argparse.ArgumentParser(description="DCT vs pixel classification comparison")
    p.add_argument("--dataset", choices=["cifar10", "stl10"], default="cifar10")
    p.add_argument("--models", default="pixelcnn,dctcnn,dctfreq",
                   help="comma list: pixelcnn,dctcnn,dctfreq,pixelresnet,dctresnet")
    p.add_argument("--mode", choices=["y_only", "ycbcr", "dc_only"], default="y_only")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--subset", type=int, default=0, help="use first N train images (0 = all)")
    p.add_argument("--augment", action="store_true", help="pixel hflip+crop vs DCT hflip")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output", "-o", default=None)
    args = p.parse_args()

    device = _get_device()
    dct_channels = {"y_only": 64, "ycbcr": 192, "dc_only": 3}[args.mode]
    print(f"Device: {device} | dataset: {args.dataset} | models: {args.models} | "
          f"mode: {args.mode} | epochs: {args.epochs} | augment: {args.augment}")

    tr_x, tr_y, te_x, te_y, num_classes, img_size = _load_pixels(args.dataset, args.data_root)
    if args.subset:
        tr_x, tr_y = tr_x[:args.subset], tr_y[:args.subset]
    block_grid = (img_size // 8, img_size // 8)
    print(f"train={len(tr_x)} test={len(te_x)} img={img_size} blocks={block_grid}")

    model_keys = args.models.split(",")
    needs_pixels = any(k.startswith("pixel") for k in model_keys)
    needs_dct = any(k.startswith("dct") for k in model_keys)

    # Test loaders (never augmented).
    te_y_t = torch.from_numpy(te_y)
    pixel_test_loader = dct_test_loader = None
    if needs_pixels:
        pixel_test_loader = DataLoader(TensorDataset(_pixel_tensors(te_x), te_y_t),
                                       batch_size=args.batch_size, num_workers=args.num_workers)
    if needs_dct:
        print("  precomputing DCT test tensors...")
        te_dct = _precompute_dct_tensors(te_x, block_grid, args.mode)
        dct_test_loader = DataLoader(TensorDataset(te_dct, te_y_t),
                                     batch_size=args.batch_size, num_workers=args.num_workers)

    # Train loaders.
    tr_y_t = torch.from_numpy(tr_y)
    if needs_dct:
        print("  precomputing DCT train tensors...")
        tr_dct = _precompute_dct_tensors(tr_x, block_grid, args.mode)

    def pixel_train_loader():
        if args.augment:
            ds = PixelAugDataset(tr_x, tr_y, img_size)
        else:
            ds = TensorDataset(_pixel_tensors(tr_x), tr_y_t)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    def dct_train_loader():
        if args.augment and args.mode == "y_only":
            ds = DCTHFlipDataset(tr_dct, tr_y_t, p=0.5)
        else:
            ds = TensorDataset(tr_dct, tr_y_t)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    results = []
    for key in model_keys:
        model = _make_model(key, num_classes, block_grid, img_size, dct_channels)
        if key.startswith("pixel"):
            r = run_experiment(key, model, pixel_train_loader(), pixel_test_loader,
                               args.epochs, device, args.lr)
        else:
            r = run_experiment(key, model, dct_train_loader(), dct_test_loader,
                               args.epochs, device, args.lr)
        r["input"] = "pixel" if key.startswith("pixel") else f"dct_{args.mode}"
        results.append(r)

    print("\n" + "=" * 78)
    print(f"{'Model':<14}{'Input':<14}{'Params':<12}{'TestAcc':<10}{'Time(s)':<10}{'Data(s)':<10}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<14}{r['input']:<14}{r['params']:<12,}"
              f"{r['final_test_accuracy']:<10.4f}{r['total_time_sec']:<10}{r['total_data_time_sec']:<10}")
    print("=" * 78)

    if args.output:
        from pathlib import Path
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "config": {
                    "dataset": args.dataset, "models": model_keys, "mode": args.mode,
                    "epochs": args.epochs, "batch_size": args.batch_size,
                    "subset": args.subset, "augment": args.augment,
                    "device": str(device),
                },
                "results": results,
            }, f, indent=2)
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
