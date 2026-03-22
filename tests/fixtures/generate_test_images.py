"""Generate synthetic test images for the test suite."""

import numpy as np
from pathlib import Path
from PIL import Image

FIXTURE_DIR = Path(__file__).parent / "test_images"


def generate_all():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Solid color (red) 64x64 ---
    solid = np.full((64, 64, 3), [255, 0, 0], dtype=np.uint8)
    Image.fromarray(solid).save(FIXTURE_DIR / "solid_red.jpg", quality=95, subsampling=0)

    # --- Gradient 128x128 ---
    gradient = np.zeros((128, 128, 3), dtype=np.uint8)
    gradient[:, :, 0] = np.linspace(0, 255, 128, dtype=np.uint8)[np.newaxis, :]
    gradient[:, :, 1] = np.linspace(0, 255, 128, dtype=np.uint8)[:, np.newaxis]
    gradient[:, :, 2] = 128
    Image.fromarray(gradient).save(FIXTURE_DIR / "gradient.jpg", quality=85)

    # --- Checkerboard 64x64 ---
    checker = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            if (i // 8 + j // 8) % 2 == 0:
                checker[i, j] = [255, 255, 255]
    Image.fromarray(checker).save(FIXTURE_DIR / "checkerboard.jpg", quality=95, subsampling=0)

    # --- Different quality factors (same source) ---
    natural = np.random.RandomState(42).randint(0, 256, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(natural)
    for q in [50, 75, 85, 95]:
        img.save(FIXTURE_DIR / f"natural_q{q}.jpg", quality=q)

    # --- Different subsampling modes ---
    img.save(FIXTURE_DIR / "sub_444.jpg", quality=85, subsampling=0)   # 4:4:4
    img.save(FIXTURE_DIR / "sub_422.jpg", quality=85, subsampling=1)   # 4:2:2
    img.save(FIXTURE_DIR / "sub_420.jpg", quality=85, subsampling=2)   # 4:2:0

    # --- Non-divisible-by-8 dimensions ---
    odd = np.random.RandomState(7).randint(0, 256, (100, 77, 3), dtype=np.uint8)
    Image.fromarray(odd).save(FIXTURE_DIR / "odd_size_100x77.jpg", quality=85)

    # --- Grayscale ---
    gray = np.random.RandomState(99).randint(0, 256, (64, 64), dtype=np.uint8)
    Image.fromarray(gray, mode="L").save(FIXTURE_DIR / "grayscale.jpg", quality=85)

    # --- PNG for converter testing ---
    Image.fromarray(natural).save(FIXTURE_DIR / "sample.png")

    # --- BMP for converter testing ---
    Image.fromarray(natural[:64, :64]).save(FIXTURE_DIR / "sample.bmp")

    # --- Small 8x8 single block ---
    tiny = np.random.RandomState(1).randint(0, 256, (8, 8, 3), dtype=np.uint8)
    Image.fromarray(tiny).save(FIXTURE_DIR / "single_block_8x8.jpg", quality=95, subsampling=0)

    print(f"Generated test images in {FIXTURE_DIR}")


if __name__ == "__main__":
    generate_all()
