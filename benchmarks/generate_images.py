"""Generate benchmark test images at various resolutions and quality factors."""

import numpy as np
from pathlib import Path
from PIL import Image

BENCH_DIR = Path(__file__).parent / "images"
RESOLUTIONS = [256, 512, 1024, 2048]
QUALITIES = [50, 75, 85, 95]


def generate():
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)

    for res in RESOLUTIONS:
        # Create a natural-looking random image (consistent across runs)
        pixels = rng.randint(0, 256, (res, res, 3), dtype=np.uint8)
        img = Image.fromarray(pixels)

        for q in QUALITIES:
            path = BENCH_DIR / f"bench_{res}x{res}_q{q}.jpg"
            img.save(str(path), quality=q, subsampling=2)

    print(f"Generated {len(RESOLUTIONS) * len(QUALITIES)} benchmark images in {BENCH_DIR}")


if __name__ == "__main__":
    generate()
