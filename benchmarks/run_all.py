"""Full benchmark suite: DCT-domain vs Pillow vs OpenCV.

Three-way comparison for each operation across resolutions and quality factors.

Usage:
    uv run python benchmarks/run_all.py
    uv run python benchmarks/run_all.py --output benchmarks/results/latest.json
    uv run python benchmarks/run_all.py --repeats 50 --resolutions 256,512
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).parent.parent))

from dct_vision.core.dct_image import DCTImage
from dct_vision.ops.blur import blur
from dct_vision.ops.sharpen import sharpen
from dct_vision.ops.color import adjust_brightness, adjust_contrast
from dct_vision.ops.scale import downscale
from dct_vision.ops.edge import detect_edges
from dct_vision.augment.flip import horizontal_flip, vertical_flip
from dct_vision.augment.crop import block_crop
from dct_vision.augment.jitter import brightness_jitter, contrast_jitter
from dct_vision.augment.noise import gaussian_noise
from dct_vision.utils.profiling import time_fn, measure_memory
from dct_vision.math.metrics import psnr, ssim

BENCH_DIR = Path(__file__).parent / "images"


# -- Pillow baselines --

def pillow_blur(pixels, sigma=2.0):
    return np.array(Image.fromarray(pixels).filter(
        ImageFilter.GaussianBlur(radius=sigma)
    ), dtype=np.uint8)

def pillow_sharpen(pixels):
    return np.array(Image.fromarray(pixels).filter(ImageFilter.SHARPEN), dtype=np.uint8)

def pillow_brightness(pixels, offset=30.0):
    return np.clip(pixels.astype(np.float32) + offset, 0, 255).astype(np.uint8)

def pillow_contrast(pixels, factor=1.5):
    mean = pixels.mean()
    return np.clip((pixels.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

def pillow_downscale(pixels, factor=2):
    img = Image.fromarray(pixels)
    return np.array(img.resize((img.width // factor, img.height // factor), Image.LANCZOS), dtype=np.uint8)

def pillow_edges(pixels):
    return np.array(Image.fromarray(pixels).convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.uint8)


# -- OpenCV baselines --

def opencv_blur(pixels, sigma=2.0):
    ksize = max(1, int(sigma * 6) | 1)  # kernel size must be odd
    return cv2.GaussianBlur(pixels, (ksize, ksize), sigma)

def opencv_sharpen(pixels):
    blurred = cv2.GaussianBlur(pixels, (0, 0), 1.5)
    return cv2.addWeighted(pixels, 1.5, blurred, -0.5, 0)

def opencv_brightness(pixels, offset=30.0):
    return np.clip(pixels.astype(np.float32) + offset, 0, 255).astype(np.uint8)

def opencv_contrast(pixels, factor=1.5):
    mean = pixels.mean()
    return np.clip((pixels.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

def opencv_downscale(pixels, factor=2):
    h, w = pixels.shape[:2]
    return cv2.resize(pixels, (w // factor, h // factor), interpolation=cv2.INTER_LANCZOS4)

def opencv_edges(pixels):
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) if pixels.ndim == 3 else pixels
    return cv2.Laplacian(gray, cv2.CV_8U)


# -- Augmentation spatial baselines --
# These simulate the full pipeline: decode -> augment -> re-encode

def pillow_hflip(pixels):
    return np.array(Image.fromarray(pixels).transpose(Image.FLIP_LEFT_RIGHT), dtype=np.uint8)

def opencv_hflip(pixels):
    return cv2.flip(pixels, 1)

def pillow_vflip(pixels):
    return np.array(Image.fromarray(pixels).transpose(Image.FLIP_TOP_BOTTOM), dtype=np.uint8)

def opencv_vflip(pixels):
    return cv2.flip(pixels, 0)

def pillow_crop(pixels):
    h, w = pixels.shape[:2]
    return pixels[:h//2, :w//2].copy()

def opencv_crop(pixels):
    h, w = pixels.shape[:2]
    return pixels[:h//2, :w//2].copy()

def pillow_noise(pixels, sigma=10.0):
    noise = np.random.RandomState(42).normal(0, sigma, pixels.shape)
    return np.clip(pixels.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def opencv_noise(pixels, sigma=10.0):
    noise = np.random.RandomState(42).normal(0, sigma, pixels.shape).astype(np.float32)
    return np.clip(pixels.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# -- Operation registry --

OPERATIONS = {
    "blur": {
        "dct":    lambda img: blur(img, sigma=2.0),
        "pillow": lambda px: pillow_blur(px, sigma=2.0),
        "opencv": lambda px: opencv_blur(px, sigma=2.0),
    },
    "sharpen": {
        "dct":    lambda img: sharpen(img, amount=1.5),
        "pillow": lambda px: pillow_sharpen(px),
        "opencv": lambda px: opencv_sharpen(px),
    },
    "brightness": {
        "dct":    lambda img: adjust_brightness(img, offset=30),
        "pillow": lambda px: pillow_brightness(px, offset=30),
        "opencv": lambda px: opencv_brightness(px, offset=30),
    },
    "contrast": {
        "dct":    lambda img: adjust_contrast(img, factor=1.5),
        "pillow": lambda px: pillow_contrast(px, factor=1.5),
        "opencv": lambda px: opencv_contrast(px, factor=1.5),
    },
    "downscale": {
        "dct":    lambda img: downscale(img, factor=2),
        "pillow": lambda px: pillow_downscale(px, factor=2),
        "opencv": lambda px: opencv_downscale(px, factor=2),
    },
    "edges": {
        "dct":    lambda img: detect_edges(img, method="laplacian"),
        "pillow": lambda px: pillow_edges(px),
        "opencv": lambda px: opencv_edges(px),
    },
    "hflip": {
        "dct":    lambda img: horizontal_flip(img),
        "pillow": lambda px: pillow_hflip(px),
        "opencv": lambda px: opencv_hflip(px),
    },
    "vflip": {
        "dct":    lambda img: vertical_flip(img),
        "pillow": lambda px: pillow_vflip(px),
        "opencv": lambda px: opencv_vflip(px),
    },
    "crop": {
        "dct":    lambda img: block_crop(img, 0, 0, img.y_coeffs.shape[0] // 2, img.y_coeffs.shape[1] // 2),
        "pillow": lambda px: pillow_crop(px),
        "opencv": lambda px: opencv_crop(px),
    },
    "noise": {
        "dct":    lambda img: gaussian_noise(img, sigma=5.0, seed=42),
        "pillow": lambda px: pillow_noise(px, sigma=10.0),
        "opencv": lambda px: opencv_noise(px, sigma=10.0),
    },
}


# -- Quality ground-truth selection --
# For each op, the spatial baseline that is the fairest mathematical equivalent,
# and whether a pixel-level quality comparison is meaningful. Random ops (noise)
# have no deterministic reference. "approx" ops use a related-but-not-identical
# spatial definition (resampling kernel, sharpen weights, edge operator) so their
# PSNR/SSIM is indicative, not exact-match.
QUALITY_REF = {
    "blur":       {"fn": lambda px: opencv_blur(px, 2.0),      "meaningful": True,  "note": "exact"},
    "sharpen":    {"fn": lambda px: opencv_sharpen(px),        "meaningful": True,  "note": "approx"},
    "brightness": {"fn": lambda px: opencv_brightness(px, 30), "meaningful": True,  "note": "exact"},
    "contrast":   {"fn": lambda px: opencv_contrast(px, 1.5),  "meaningful": True,  "note": "exact"},
    "downscale":  {"fn": lambda px: opencv_downscale(px, 2),   "meaningful": True,  "note": "approx"},
    "edges":      {"fn": lambda px: opencv_edges(px),          "meaningful": True,  "note": "approx"},
    "hflip":      {"fn": lambda px: opencv_hflip(px),          "meaningful": True,  "note": "exact"},
    "vflip":      {"fn": lambda px: opencv_vflip(px),          "meaningful": True,  "note": "exact"},
    "crop":       {"fn": lambda px: opencv_crop(px),           "meaningful": True,  "note": "exact"},
    "noise":      {"fn": None,                                 "meaningful": False, "note": "random"},
}


def _align(a, b):
    """Align two images for quality comparison: match channels and spatial size."""
    if a.ndim != b.ndim:
        if a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        if b.ndim == 3:
            b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def _compute_quality(op_name, dct_img, pixels):
    """PSNR/SSIM of DCT-domain output vs the spatial ground truth."""
    ref = QUALITY_REF.get(op_name, {})
    if not ref.get("meaningful") or ref.get("fn") is None:
        return {"psnr_db": None, "ssim": None, "quality_note": ref.get("note", "n/a")}
    try:
        dct_out = OPERATIONS[op_name]["dct"](dct_img).to_pixels()
        gt = ref["fn"](pixels)
        a, b = _align(dct_out, gt)
        p = psnr(a, b)
        return {
            "psnr_db": (round(p, 2) if p != float("inf") else "inf"),
            "ssim": round(ssim(a, b), 4),
            "quality_note": ref["note"],
        }
    except Exception as exc:  # noqa: BLE001 - benchmark robustness
        return {"psnr_db": None, "ssim": None, "quality_note": f"error: {exc}"}


def benchmark_operation(op_name, jpeg_path, warmup=3, repeats=20):
    """Benchmark one operation: DCT vs Pillow vs OpenCV (op-only timing + quality + memory)."""
    op = OPERATIONS[op_name]

    pixels = np.array(Image.open(jpeg_path), dtype=np.uint8)
    dct_img = DCTImage.from_file(jpeg_path)

    # Time each path (operation only, image already loaded)
    dct_timing = time_fn(lambda: op["dct"](dct_img), warmup=warmup, repeats=repeats)
    pillow_timing = time_fn(lambda: op["pillow"](pixels), warmup=warmup, repeats=repeats)
    opencv_timing = time_fn(lambda: op["opencv"](pixels), warmup=warmup, repeats=repeats)

    # Speedup ratios (vs DCT)
    pillow_vs_dct = pillow_timing["mean_ms"] / max(dct_timing["mean_ms"], 0.001)
    opencv_vs_dct = opencv_timing["mean_ms"] / max(dct_timing["mean_ms"], 0.001)

    # Peak memory of the operation itself (single call under tracemalloc).
    dct_mem = measure_memory(lambda: op["dct"](dct_img))["peak_kb"]
    pixel_mem = measure_memory(lambda: op["opencv"](pixels))["peak_kb"]

    quality = _compute_quality(op_name, dct_img, pixels)

    return {
        "operation": op_name,
        "dct_ms": round(dct_timing["mean_ms"], 3),
        "dct_std": round(dct_timing["std_ms"], 3),
        "pillow_ms": round(pillow_timing["mean_ms"], 3),
        "pillow_std": round(pillow_timing["std_ms"], 3),
        "opencv_ms": round(opencv_timing["mean_ms"], 3),
        "opencv_std": round(opencv_timing["std_ms"], 3),
        "speedup_vs_pillow": round(pillow_vs_dct, 2),
        "speedup_vs_opencv": round(opencv_vs_dct, 2),
        "dct_peak_kb": round(dct_mem, 1),
        "pixel_peak_kb": round(pixel_mem, 1),
        "mem_ratio_vs_pixel": round(pixel_mem / max(dct_mem, 0.001), 2),
        **quality,
    }


def _dct_end_to_end(jpeg_path, op_name, out_path):
    """Full DCT pipeline: extract coefficients -> op -> write JPEG (no full decode).

    Uses quality=None so save() takes the native lossless coefficient-transcode
    path (the actual DCT advantage) instead of re-encoding from pixels. Ops that
    change dimensions (downscale, crop) fall back to a pixel encode automatically.
    """
    img = DCTImage.from_file(jpeg_path)
    img = OPERATIONS[op_name]["dct"](img)
    img.save(out_path, quality=None)


def _pixel_end_to_end(jpeg_path, op_name, out_path):
    """Full pixel pipeline: decode -> op -> encode."""
    px = np.array(Image.open(jpeg_path), dtype=np.uint8)
    out = OPERATIONS[op_name]["opencv"](px)
    if out.ndim == 2:
        Image.fromarray(out).save(out_path, quality=85)
    else:
        Image.fromarray(out).save(out_path, quality=85)


def benchmark_end_to_end(op_name, jpeg_path, warmup=2, repeats=10):
    """End-to-end throughput: load + op + save. This is where skipping decode pays off."""
    with tempfile.TemporaryDirectory() as td:
        dct_out = os.path.join(td, "dct.jpg")
        px_out = os.path.join(td, "px.jpg")

        dct_timing = time_fn(
            lambda: _dct_end_to_end(jpeg_path, op_name, dct_out),
            warmup=warmup, repeats=repeats,
        )
        pixel_timing = time_fn(
            lambda: _pixel_end_to_end(jpeg_path, op_name, px_out),
            warmup=warmup, repeats=repeats,
        )

    speedup = pixel_timing["mean_ms"] / max(dct_timing["mean_ms"], 0.001)
    return {
        "operation": op_name,
        "dct_e2e_ms": round(dct_timing["mean_ms"], 3),
        "dct_e2e_std": round(dct_timing["std_ms"], 3),
        "pixel_e2e_ms": round(pixel_timing["mean_ms"], 3),
        "pixel_e2e_std": round(pixel_timing["std_ms"], 3),
        "e2e_speedup": round(speedup, 2),
    }


def run_benchmarks(resolutions=None, qualities=None, operations=None, warmup=3,
                   repeats=20, e2e=True, e2e_repeats=10):
    if resolutions is None:
        resolutions = [256, 512, 1024]
    if qualities is None:
        qualities = [50, 85, 95]
    if operations is None:
        operations = list(OPERATIONS.keys())

    results = []
    e2e_results = []
    total = len(operations) * len(resolutions) * len(qualities)
    count = 0

    for op_name in operations:
        for res in resolutions:
            for q in qualities:
                count += 1
                image_path = BENCH_DIR / f"bench_{res}x{res}_q{q}.jpg"
                if not image_path.exists():
                    print(f"  [{count}/{total}] SKIP {op_name} {res}x{res} q{q}")
                    continue

                print(f"  [{count}/{total}] {op_name} {res}x{res} q{q} ...", end=" ", flush=True)
                r = benchmark_operation(op_name, str(image_path), warmup=warmup, repeats=repeats)
                r["resolution"] = res
                r["quality"] = q
                results.append(r)

                q_str = f"PSNR={r['psnr_db']}" if r["psnr_db"] is not None else "PSNR=n/a"
                print(
                    f"DCT={r['dct_ms']:.1f}ms  "
                    f"OpenCV={r['opencv_ms']:.1f}ms  "
                    f"vs_Pillow={r['speedup_vs_pillow']:.1f}x  "
                    f"vs_OpenCV={r['speedup_vs_opencv']:.1f}x  "
                    f"{q_str}  mem={r['mem_ratio_vs_pixel']:.1f}x"
                )

                if e2e:
                    er = benchmark_end_to_end(op_name, str(image_path), repeats=e2e_repeats)
                    er["resolution"] = res
                    er["quality"] = q
                    e2e_results.append(er)
                    print(
                        f"        e2e: DCT={er['dct_e2e_ms']:.1f}ms  "
                        f"Pixel={er['pixel_e2e_ms']:.1f}ms  "
                        f"speedup={er['e2e_speedup']:.1f}x"
                    )

    return results, e2e_results


def _fmt(v):
    return "n/a" if v is None else str(v)


def print_summary(results, e2e_results=None):
    hdr = (
        f"{'Operation':<12} {'Res':<6} "
        f"{'DCT (ms)':<10} {'OpenCV (ms)':<12} "
        f"{'vs Pillow':<10} {'vs OpenCV':<10} {'PSNR':<8} {'SSIM':<8} {'mem x':<7}"
    )
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        print(
            f"{r['operation']:<12} {r['resolution']:<6} "
            f"{r['dct_ms']:<10.2f} {r['opencv_ms']:<12.2f} "
            f"{r['speedup_vs_pillow']:<10.1f}x {r['speedup_vs_opencv']:<10.1f}x "
            f"{_fmt(r.get('psnr_db')):<8} {_fmt(r.get('ssim')):<8} "
            f"{_fmt(r.get('mem_ratio_vs_pixel')):<7}"
        )

    print("=" * len(hdr))

    # Averages by operation
    print("\n--- Op-only averages by Operation ---")
    print(f"{'Operation':<12} {'DCT (ms)':<10} {'OpenCV (ms)':<12} "
          f"{'vs Pillow':<10} {'vs OpenCV':<10}")
    print("-" * 56)
    ops = sorted(set(r["operation"] for r in results))
    for op in ops:
        op_r = [r for r in results if r["operation"] == op]
        avg_dct = np.mean([r["dct_ms"] for r in op_r])
        avg_cv = np.mean([r["opencv_ms"] for r in op_r])
        avg_vs_pil = np.mean([r["speedup_vs_pillow"] for r in op_r])
        avg_vs_cv = np.mean([r["speedup_vs_opencv"] for r in op_r])
        print(
            f"{op:<12} {avg_dct:<10.2f} {avg_cv:<12.2f} "
            f"{avg_vs_pil:<10.1f}x {avg_vs_cv:<10.1f}x"
        )

    if e2e_results:
        print("\n--- End-to-end (load + op + save) averages by Operation ---")
        print(f"{'Operation':<12} {'DCT e2e (ms)':<14} {'Pixel e2e (ms)':<16} {'Speedup':<10}")
        print("-" * 52)
        for op in sorted(set(r["operation"] for r in e2e_results)):
            op_r = [r for r in e2e_results if r["operation"] == op]
            avg_dct = np.mean([r["dct_e2e_ms"] for r in op_r])
            avg_px = np.mean([r["pixel_e2e_ms"] for r in op_r])
            avg_sp = np.mean([r["e2e_speedup"] for r in op_r])
            print(f"{op:<12} {avg_dct:<14.2f} {avg_px:<16.2f} {avg_sp:<10.1f}x")


def main():
    parser = argparse.ArgumentParser(description="DCT-Vision benchmark suite (3-way)")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--resolutions", type=str, default="256,512,1024,2048")
    parser.add_argument("--qualities", type=str, default="50,75,85,95")
    parser.add_argument("--operations", type=str, default=None)
    parser.add_argument("--no-e2e", action="store_true", help="Skip end-to-end pipeline benchmark")
    parser.add_argument("--e2e-repeats", type=int, default=10)
    args = parser.parse_args()

    resolutions = [int(x) for x in args.resolutions.split(",")]
    qualities = [int(x) for x in args.qualities.split(",")]
    operations = args.operations.split(",") if args.operations else None

    print("DCT-Vision Benchmark Suite (DCT vs Pillow vs OpenCV)")
    print(f"Resolutions: {resolutions}")
    print(f"Qualities: {qualities}")
    print(f"Repeats: {args.repeats}, Warmup: {args.warmup}")
    print()

    missing = any(
        not (BENCH_DIR / f"bench_{r}x{r}_q{q}.jpg").exists()
        for r in resolutions for q in qualities
    )
    if missing:
        print("Generating benchmark images...")
        from benchmarks.generate_images import generate
        generate()
        print()

    results, e2e_results = run_benchmarks(
        resolutions=resolutions,
        qualities=qualities,
        operations=operations,
        warmup=args.warmup,
        repeats=args.repeats,
        e2e=not args.no_e2e,
        e2e_repeats=args.e2e_repeats,
    )

    print_summary(results, e2e_results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "metadata": {
                    "repeats": args.repeats,
                    "warmup": args.warmup,
                    "resolutions": resolutions,
                    "qualities": qualities,
                    "metrics": ["op_ms", "psnr_db", "ssim", "peak_kb", "e2e_ms"],
                },
                "results": results,
                "end_to_end": e2e_results,
            }, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
