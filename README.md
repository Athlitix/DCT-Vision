# DCT-Vision

Frequency-domain native image processing. Operates directly on JPEG DCT coefficients, skipping the pixel decode step entirely.

## Why?

JPEG images are already stored as DCT coefficients. Decoding to pixels just to blur/sharpen/adjust is wasteful. DCT-Vision works directly on the coefficients -- many operations become simple multiplications instead of expensive convolutions.

## Performance

Full methodology and raw numbers: [`benchmarks/results/RESULTS.md`](benchmarks/results/RESULTS.md)
(100 reps, resolutions 256-2048, JPEG quality 50-95). Reproduce with `make bench`.

**Operation-only speed and quality** (averaged over all resolutions/qualities).
`PSNR`/`SSIM` compare the DCT op against the same operation applied in the
spatial domain, so they measure the operation's own error. `inf`/`1.000` means
the transform is exact (lossless).

| Operation | DCT time | vs Pillow | vs OpenCV | PSNR (dB) | SSIM |
|-----------|---------:|----------:|----------:|----------:|-----:|
| Brightness | 0.34ms | 18.6x | 18.2x | 43.2 | 1.000 |
| Contrast   | 1.42ms | 18.2x | 18.3x | 23.0 | 0.972 |
| Sharpen    | 1.50ms | 18.9x | 1.6x  | 23.0 | 0.964 |
| Blur       | 2.06ms | 18.4x | 0.9x  | 26.8 | 0.570 |
| Edge detect| 4.21ms | 12.2x | 0.7x  | 8.6  | 0.332 |
| Noise      | 24.2ms | 3.3x  | 3.1x  | -    | -    |
| Downscale 2x| 4.98ms| 3.6x  | 1.1x  | 14.1 | 0.406 |
| Rotate 90  | 1.78ms | 1.6x  | 0.4x  | 25.6 | 0.972 |
| H-flip     | 0.75ms | 2.7x  | 0.3x  | inf  | 1.000 |
| V-flip     | 1.12ms | 1.5x  | 0.1x  | inf  | 1.000 |
| Crop       | 0.04ms | 0.8x  | 0.8x  | inf  | 1.000 |

**End-to-end** (load + operation + save vs decode + operation + encode) is the
fairer headline, since the whole premise is skipping the decode. DCT writes
modified coefficients back losslessly:

| Operation | DCT e2e speedup vs pixel pipeline |
|-----------|:--------------------------------:|
| Noise     | 1.84x |
| Contrast  | 1.55x |
| Brightness| 1.03x |
| Sharpen / flips / rotate | ~0.7-0.9x |
| Downscale / crop / edges | 0.1-0.6x |

### Honest summary

- **Big, consistent wins vs Pillow** on pointwise ops (brightness, contrast,
  sharpen, blur, edges): 12-19x.
- **vs OpenCV is mixed and we say so.** Brightness/contrast/noise beat OpenCV
  several-fold; blur is about par; OpenCV's SIMD wins on flips, rotate, crop,
  downscale and edges. We do not claim to beat OpenCV at everything.
- **Exact operations** (brightness, flips, crop, and 90/180/270 rotation on
  4:4:4 images) are provably lossless. Blur/downscale/edges are approximations
  with the quality shown above.
- **End-to-end** favors DCT on expensive ops and large images; on cheap ops it
  is roughly par, because libjpeg-turbo decode is already fast, so there is less
  to save. The advantage compounds in batch pipelines that avoid repeated decode.

## Install

```bash
pip install dct-vision
```

## Quick start

### Python API

```python
from dct_vision.core.dct_image import DCTImage
from dct_vision.ops.blur import blur
from dct_vision.ops.color import adjust_brightness

# Load JPEG (extracts DCT coefficients directly, no pixel decode)
img = DCTImage.from_file("photo.jpg")

# Process in frequency domain
img = blur(img, sigma=2.0)
img = adjust_brightness(img, offset=20)

# Save (writes coefficients directly, no pixel encode)
img.save("output.jpg")
```

### CLI

```bash
dv blur photo.jpg -o blurred.jpg --sigma 2.0
dv sharpen photo.jpg -o sharp.jpg --amount 1.5
dv brightness photo.jpg -o bright.jpg --offset 30
dv contrast photo.jpg -o contrast.jpg --factor 1.5
dv downscale photo.jpg -o small.jpg --factor 2
dv edges photo.jpg -o edges.jpg --method laplacian
dv rotate photo.jpg -o rotated.jpg --degrees 90     # lossless
dv info photo.jpg --json
dv quality photo.jpg
dv convert input.png -o output.jpg --quality 85
dv augment photo.jpg -o aug.jpg --flip horizontal --noise 3.0 --seed 42

# Applications
dv apps thumbnail photo.jpg -o thumb.jpg --size 64  # from DC coeffs, no IDCT
dv apps dedup ./photos/ --max-distance 5            # find near-duplicates
dv apps forensics photo.jpg                         # double-compression check
```

### ML Augmentation Pipeline

```python
from dct_vision.core.dct_image import DCTImage
from dct_vision.augment.flip import horizontal_flip
from dct_vision.augment.jitter import brightness_jitter
from dct_vision.augment.noise import gaussian_noise

img = DCTImage.from_file("train/img_001.jpg")
img = horizontal_flip(img)
img = brightness_jitter(img, max_offset=20, seed=42)
img = gaussian_noise(img, sigma=2.0, seed=42)
img.save("augmented/img_001.jpg")
```

## Operations

| Operation | Type | How it works |
|-----------|------|-------------|
| Gaussian blur | Tier 1/2 | Multiply coefficients by Gaussian envelope (cross-block for sigma > 2) |
| Sharpening | Tier 1 | Boost high-frequency coefficients |
| Brightness | Tier 1 | Offset DC coefficient (block mean) |
| Contrast | Tier 1 | Scale AC coefficients (deviation from mean) |
| Downscale 2x | Tier 1 | Merge 2x2 block groups via transform matrix |
| Edge detection | Tier 2 | Laplacian or gradient in frequency domain |
| Sobel edge detection | Tier 1 | Directional frequency gradient weights |
| Scharr edge detection | Tier 1 | Weighted directional gradient (more accurate) |
| Box blur | Tier 1 | Sinc-like frequency envelope |
| Emboss | Tier 1 | Directional frequency emphasis |
| Band-pass filter | Tier 1 | Keep mid-frequency coefficients (no OpenCV equivalent) |
| Unsharp mask | Tier 1 | 1 + amount * (1 - Gaussian envelope) |
| Color temperature | Tier 1 | Shift Cb/Cr DC coefficients |
| Saturation | Tier 1 | Scale Cb/Cr coefficients |
| Wiener denoising | Tier 1 | Optimal frequency-domain noise filter |
| JPEG deblocking | Tier 1 | Attenuate high-freq quantization artifacts |
| Perceptual hash (pHash) | Tier 1 | Hash from DC coefficients (native DCT advantage) |
| Blur detection | Analysis | High-freq to total energy ratio |
| Noise estimation | Analysis | Std of highest-frequency coefficients |
| Texture complexity | Analysis | Nonzero AC coefficient ratio |
| Image similarity | Analysis | Normalized cross-correlation of coefficients |
| Vignette | Photo | Distance-weighted block attenuation |
| Sepia / tint | Photo | Set Cb/Cr to fixed warm values |
| Grayscale conversion | Photo | Drop Cb/Cr channels (zero cost) |
| Posterize | Photo | Aggressive coefficient requantization |
| Solarize | Photo | Invert coefficients above threshold |
| Requantize (change JPEG quality) | Compression | Apply new quant table without decode |
| Coefficient pruning | Compression | Zero small AC coefficients to reduce file size |
| Quality estimation | Tier 1 | Reverse-engineer quality from quant tables |
| Horizontal/vertical flip | Augment | Negate odd-indexed frequency coefficients |
| Rotate 90/180/270 + transpose | Geometry | Lossless coefficient permutation (jpegtran-style, exact) |
| Block crop | Augment | Slice coefficient array directly |
| Brightness/contrast jitter | Augment | Random DC/AC perturbation |
| Gaussian noise | Augment | Add noise to AC coefficients |

## PyTorch Integration

```python
from dct_vision.ml.dataset import DCTDataset
from torch.utils.data import DataLoader

dataset = DCTDataset("train/", mode="y_only", resize_blocks=(4, 4),
                     augmentations=["hflip:p=0.5", "noise:sigma=2.0"])
loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

Pre-cache for instant loading:
```bash
dv dataset prepare ./train/ -o ./train_dct/
dv dataset bench ./train/ --mode y_only --batch-size 32
```

### DCT-input classification (CIFAR-10)

DCT coefficients feed straight into a CNN/ResNet (Uber 2018 style). The network
sees a 4x4 block grid instead of a 32x32 image, so it does far less spatial
compute. Three inputs compared fairly (same architecture, same augmentation):
RGB pixels, DCT Y-only (grayscale), and DCT YCbCr (full color).

| Input | Simple CNN acc | ResNet-18 acc | Train time (ResNet) |
|-------|---------------:|--------------:|--------------------:|
| RGB pixels       | 0.833 | 0.896 | 724 s |
| DCT Y-only       | 0.579 | 0.638 | 88 s (8.2x faster) |
| DCT YCbCr (color)| 0.648 | 0.676 | 97 s (7.5x faster) |

**Reading it honestly:** DCT input gives a large, consistent **training speedup
(~7-8x)** but does not match pixel accuracy on CIFAR-10. Adding color (YCbCr)
recovers 4-7 points over Y-only, confirming that part of the earlier Y-only gap
was a color handicap, not the representation. The remaining gap is expected:
CIFAR images are only 32x32, so DCT's 8x block downsampling leaves a 4x4 grid
and discards spatial detail. The speedup and the accuracy cost are two ends of
the same dial (less spatial resolution = less compute). Uber's accuracy-parity
result was on ImageNet, where 8x downsampling still leaves ample resolution;
closing the gap here needs larger inputs (STL-10/ImageNet), which the harness
supports (`--dataset stl10`).

## Applications

```python
from dct_vision.apps import find_duplicates, detect_double_compression, dc_thumbnail
from dct_vision.core.dct_image import DCTImage

# Near-duplicate detection over a folder (pHash on coefficients, no decode)
groups = find_duplicates("./photos/", max_distance=5)

# Instant thumbnail from DC coefficients (one array op, no IDCT)
thumb = dc_thumbnail(DCTImage.from_file("photo.jpg"), size=64)

# JPEG double-compression forensics
report = detect_double_compression(DCTImage.from_file("suspect.jpg"))
```

## Documentation

- [Installation guide](docs/installation.md)
- [CLI reference](docs/cli-reference.md)
- [Python API](docs/python-api.md)
- [Architecture](docs/architecture.md)
- [ML integration](docs/ml-integration.md)

## Requirements

- Python 3.10+
- libjpeg-turbo (for native DCT extraction; falls back to Pillow if unavailable)

## License

MIT
