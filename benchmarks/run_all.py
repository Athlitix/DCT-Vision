"""Full benchmark suite: DCT-domain vs Pillow vs OpenCV.

Three-way comparison for each operation across resolutions and quality factors.

Usage:
    uv run python benchmarks/run_all.py
    uv run python benchmarks/run_all.py --output benchmarks/results/latest.json
    uv run python benchmarks/run_all.py --repeats 50 --resolutions 256,512
"""

import argparse
import json
import sys
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
from dct_vision.utils.profiling import time_fn, psnr

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
}


def benchmark_operation(op_name, jpeg_path, warmup=3, repeats=20):
    """Benchmark one operation: DCT vs Pillow vs OpenCV."""
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
    }


def run_benchmarks(resolutions=None, qualities=None, operations=None, warmup=3, repeats=20):
    if resolutions is None:
        resolutions = [256, 512, 1024]
    if qualities is None:
        qualities = [50, 85, 95]
    if operations is None:
        operations = list(OPERATIONS.keys())

    results = []
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
                print(
                    f"DCT={r['dct_ms']:.1f}ms  "
                    f"Pillow={r['pillow_ms']:.1f}ms  "
                    f"OpenCV={r['opencv_ms']:.1f}ms  "
                    f"vs_Pillow={r['speedup_vs_pillow']:.1f}x  "
                    f"vs_OpenCV={r['speedup_vs_opencv']:.1f}x"
                )

    return results


def print_summary(results):
    hdr = (
        f"{'Operation':<12} {'Res':<6} "
        f"{'DCT (ms)':<10} {'Pillow (ms)':<12} {'OpenCV (ms)':<12} "
        f"{'vs Pillow':<10} {'vs OpenCV':<10}"
    )
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        print(
            f"{r['operation']:<12} {r['resolution']:<6} "
            f"{r['dct_ms']:<10.2f} {r['pillow_ms']:<12.2f} {r['opencv_ms']:<12.2f} "
            f"{r['speedup_vs_pillow']:<10.1f}x {r['speedup_vs_opencv']:<10.1f}x"
        )

    print("=" * len(hdr))

    # Averages by operation
    print("\n--- Averages by Operation ---")
    print(f"{'Operation':<12} {'DCT (ms)':<10} {'Pillow (ms)':<12} {'OpenCV (ms)':<12} "
          f"{'vs Pillow':<10} {'vs OpenCV':<10}")
    print("-" * 66)
    ops = sorted(set(r["operation"] for r in results))
    for op in ops:
        op_r = [r for r in results if r["operation"] == op]
        avg_dct = np.mean([r["dct_ms"] for r in op_r])
        avg_pil = np.mean([r["pillow_ms"] for r in op_r])
        avg_cv = np.mean([r["opencv_ms"] for r in op_r])
        avg_vs_pil = np.mean([r["speedup_vs_pillow"] for r in op_r])
        avg_vs_cv = np.mean([r["speedup_vs_opencv"] for r in op_r])
        print(
            f"{op:<12} {avg_dct:<10.2f} {avg_pil:<12.2f} {avg_cv:<12.2f} "
            f"{avg_vs_pil:<10.1f}x {avg_vs_cv:<10.1f}x"
        )


def main():
    parser = argparse.ArgumentParser(description="DCT-Vision benchmark suite (3-way)")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--resolutions", type=str, default="256,512,1024")
    parser.add_argument("--qualities", type=str, default="50,85,95")
    parser.add_argument("--operations", type=str, default=None)
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

    results = run_benchmarks(
        resolutions=resolutions,
        qualities=qualities,
        operations=operations,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    print_summary(results)

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
                },
                "results": results,
            }, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
