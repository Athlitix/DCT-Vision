"""Cross-block seam analysis: DCT blur with vs without the overlapping-block
strategy, swept over Gaussian sigma.

For each sigma we compare three things against a true full-image spatial
Gaussian blur (the ground truth):

  A. per-block DCT blur   (cross_block=False) -- blocks blurred independently,
     expected to show seams at 8x8 boundaries for larger sigma.
  B. cross-block DCT blur (cross_block=True)  -- 3x3 neighborhood strategy.

Metrics per method:
  - PSNR / SSIM vs the true spatial blur (higher = closer).
  - blockiness: average pixel jump across 8x8 block boundaries minus the
    average jump at interior positions (higher = more visible seams).

Usage:
    uv run python benchmarks/cross_block_sweep.py
    uv run python benchmarks/cross_block_sweep.py -o benchmarks/results/cross_block.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from dct_vision.core.dct_image import DCTImage
from dct_vision.ops.blur import blur
from dct_vision.math.metrics import psnr, ssim
from dct_vision.utils.profiling import time_fn

BENCH_DIR = Path(__file__).parent / "images"
SIGMAS = [0.5, 1.0, 2.0, 4.0, 8.0]


def _to_gray(px):
    return cv2.cvtColor(px, cv2.COLOR_RGB2GRAY) if px.ndim == 3 else px


def blockiness(gray, block=8):
    """Average pixel jump across block boundaries minus interior jumps.

    A value near 0 means boundaries are indistinguishable from interior (no
    seams); larger positive values mean visible blocking.
    """
    g = gray.astype(np.float64)
    # Vertical boundaries: columns that are multiples of `block` (excluding 0).
    diff_h = np.abs(g[:, 1:] - g[:, :-1])  # jump at column j sits at index j-1
    cols = np.arange(1, g.shape[1])
    is_boundary = (cols % block == 0)
    boundary_v = diff_h[:, is_boundary].mean() if is_boundary.any() else 0.0
    interior_v = diff_h[:, ~is_boundary].mean() if (~is_boundary).any() else 0.0

    diff_v = np.abs(g[1:, :] - g[:-1, :])
    rows = np.arange(1, g.shape[0])
    is_boundary_r = (rows % block == 0)
    boundary_h = diff_v[is_boundary_r, :].mean() if is_boundary_r.any() else 0.0
    interior_h = diff_v[~is_boundary_r, :].mean() if (~is_boundary_r).any() else 0.0

    return float(((boundary_v - interior_v) + (boundary_h - interior_h)) / 2.0)


def sweep(jpeg_path, repeats=10):
    img = DCTImage.from_file(jpeg_path)
    base_px = img.to_pixels()
    base_gray = _to_gray(base_px)
    base_block = blockiness(base_gray)

    rows = []
    for sigma in SIGMAS:
        # Ground truth: true spatial Gaussian blur of the decoded image.
        ksize = max(1, int(sigma * 6) | 1)
        gt = cv2.GaussianBlur(base_px, (ksize, ksize), sigma)
        gt_gray = _to_gray(gt)

        row = {"sigma": sigma, "gt_blockiness": round(base_block, 3)}

        for label, cross in (("per_block", False), ("cross_block", True)):
            out = blur(img, sigma=sigma, cross_block=cross).to_pixels()
            og, gg = out, gt
            if og.shape != gg.shape:
                og, gg = _to_gray(og), gt_gray
            p = psnr(og, gg)
            timing = time_fn(lambda: blur(img, sigma=sigma, cross_block=cross), warmup=2, repeats=repeats)
            row[f"{label}_psnr"] = round(p, 2) if p != float("inf") else "inf"
            row[f"{label}_ssim"] = round(ssim(_to_gray(out), gt_gray), 4)
            row[f"{label}_blockiness"] = round(blockiness(_to_gray(out)), 3)
            row[f"{label}_ms"] = round(timing["mean_ms"], 3)

        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Cross-block seam sigma sweep")
    ap.add_argument("--image", default=str(BENCH_DIR / "bench_512x512_q95.jpg"))
    ap.add_argument("--output", "-o", default=None)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    if not Path(args.image).exists():
        from benchmarks.generate_images import generate
        generate()

    print(f"Cross-block seam sweep on {Path(args.image).name}")
    rows = sweep(args.image, repeats=args.repeats)

    hdr = (f"{'sigma':<7}{'perblk PSNR':<13}{'xblk PSNR':<12}"
           f"{'perblk seam':<13}{'xblk seam':<12}{'perblk ms':<11}{'xblk ms':<10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['sigma']:<7}{str(r['per_block_psnr']):<13}{str(r['cross_block_psnr']):<12}"
              f"{r['per_block_blockiness']:<13}{r['cross_block_blockiness']:<12}"
              f"{r['per_block_ms']:<11}{r['cross_block_ms']:<10}")

    # Determine the seam threshold: smallest sigma where per-block blockiness
    # exceeds the ground-truth baseline by more than 1.0 pixel.
    threshold = None
    for r in rows:
        if r["per_block_blockiness"] - r["gt_blockiness"] > 1.0:
            threshold = r["sigma"]
            break
    print(f"\nPer-block seams become visible (blockiness > gt+1.0) at sigma >= "
          f"{threshold if threshold is not None else 'none in range'}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "image": Path(args.image).name,
                "sigmas": SIGMAS,
                "seam_threshold_sigma": threshold,
                "results": rows,
            }, f, indent=2)
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
