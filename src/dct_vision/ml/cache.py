"""DCT coefficient caching for fast dataset loading.

Pre-extracts DCT coefficients from JPEG files and stores them
as .npz files. Subsequent loads skip libjpeg entirely.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from dct_vision.core.dct_image import DCTImage


def prepare_cache(
    src_dir: str,
    dst_dir: str,
    quality: int = 85,
) -> dict:
    """Pre-extract DCT coefficients from all images in src_dir.

    Parameters
    ----------
    src_dir : str
        Source directory (ImageFolder-style or flat).
    dst_dir : str
        Output directory for .npz files (mirrors src structure).
    quality : int
        Quality for non-JPEG conversion.

    Returns
    -------
    dict
        Stats: count, total_blocks, total_bytes.
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    count = 0
    total_blocks = 0

    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        out_dir = dst / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        for fname in sorted(files):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            in_path = Path(root) / fname
            out_path = out_dir / (Path(fname).stem + ".npz")

            try:
                if str(in_path).lower().endswith((".jpg", ".jpeg")):
                    img = DCTImage.from_file(str(in_path))
                else:
                    from dct_vision.io.convert import convert_to_dct
                    img = convert_to_dct(str(in_path), quality=quality)

                data = {
                    "y_coeffs": img.y_coeffs,
                    "width": np.array(img.width),
                    "height": np.array(img.height),
                }
                for i, qt in enumerate(img.quant_tables):
                    data[f"quant_{i}"] = qt

                if img.cb_coeffs is not None:
                    data["cb_coeffs"] = img.cb_coeffs
                    data["cr_coeffs"] = img.cr_coeffs

                np.savez_compressed(str(out_path), **data)
                count += 1
                total_blocks += img.y_coeffs.shape[0] * img.y_coeffs.shape[1]

            except Exception as e:
                print(f"  Skipping {in_path}: {e}")

    total_bytes = sum(f.stat().st_size for f in dst.rglob("*.npz"))

    return {
        "count": count,
        "total_blocks": total_blocks,
        "total_bytes": total_bytes,
    }


def load_cached(path: str) -> DCTImage:
    """Load a cached .npz file as DCTImage."""
    data = np.load(path)
    y_coeffs = data["y_coeffs"]
    width = int(data["width"])
    height = int(data["height"])

    quant_tables = []
    for i in range(4):
        key = f"quant_{i}"
        if key in data:
            quant_tables.append(data[key])

    cb_coeffs = data["cb_coeffs"] if "cb_coeffs" in data else None
    cr_coeffs = data["cr_coeffs"] if "cr_coeffs" in data else None

    comp_info = None
    if cb_coeffs is not None:
        bh, bw = y_coeffs.shape[:2]
        ch_bh = cb_coeffs.shape[0]
        h_samp = bh // ch_bh if ch_bh > 0 else 1
        v_samp = bw // cb_coeffs.shape[1] if cb_coeffs.shape[1] > 0 else 1
        comp_info = [
            {"h_samp_factor": h_samp, "v_samp_factor": v_samp, "quant_tbl_no": 0},
            {"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": min(1, len(quant_tables) - 1)},
            {"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": min(1, len(quant_tables) - 1)},
        ]

    return DCTImage(
        y_coeffs=y_coeffs,
        cb_coeffs=cb_coeffs,
        cr_coeffs=cr_coeffs,
        quant_tables=quant_tables,
        width=width,
        height=height,
        comp_info=comp_info,
    )


def dataset_info(root: str) -> dict:
    """Get stats about a dataset directory."""
    root_path = Path(root)
    npz_files = list(root_path.rglob("*.npz"))
    jpg_files = list(root_path.rglob("*.jpg")) + list(root_path.rglob("*.jpeg"))
    png_files = list(root_path.rglob("*.png"))

    classes = sorted([
        d.name for d in root_path.iterdir()
        if d.is_dir() and any(d.iterdir())
    ])

    total_images = len(npz_files) + len(jpg_files) + len(png_files)
    total_bytes = sum(f.stat().st_size for f in root_path.rglob("*") if f.is_file())

    return {
        "root": str(root),
        "total_images": total_images,
        "jpeg_count": len(jpg_files),
        "npz_count": len(npz_files),
        "png_count": len(png_files),
        "classes": len(classes),
        "class_names": classes,
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
    }
