# DCT-Vision

Frequency-domain native image processing. Operates directly on JPEG DCT coefficients, skipping the pixel decode step entirely.

## Why?

JPEG images are already stored as DCT coefficients. Decoding to pixels just to blur/sharpen/adjust is wasteful. DCT-Vision works directly on the coefficients -- many operations become simple multiplications instead of expensive convolutions.

## Performance

Benchmarked on 1024x1024 JPEG (operation time only, image already loaded):

| Operation | DCT-Vision | Pillow | OpenCV | vs Pillow | vs OpenCV |
|-----------|-----------|--------|--------|-----------|-----------|
| Blur | 2.0ms | 21.1ms | 1.2ms | 10.5x | 0.6x |
| Sharpen | 1.9ms | 19.3ms | 2.9ms | 10.1x | 1.5x |
| Brightness | 0.2ms | 5.3ms | 5.6ms | 26.5x | 28.0x |
| Contrast | 0.6ms | 14.5ms | 15.4ms | 24.2x | 25.7x |
| Noise | 13.6ms | 58.4ms | 53.5ms | 4.3x | 3.9x |
| Edge detect | 0.8ms | 11.6ms | 0.9ms | 14.5x | 1.1x |

Full pipeline (load + flip + brightness + noise + save, 1024x1024):
- DCT-Vision: **83ms** | Pillow: 117ms | OpenCV: 107ms

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
dv info photo.jpg --json
dv quality photo.jpg
dv convert input.png -o output.jpg --quality 85
dv augment photo.jpg -o aug.jpg --flip horizontal --noise 3.0 --seed 42
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
| Block crop | Augment | Slice coefficient array directly |
| Brightness/contrast jitter | Augment | Random DC/AC perturbation |
| Gaussian noise | Augment | Add noise to AC coefficients |

## Documentation

- [Installation guide](docs/installation.md)
- [CLI reference](docs/cli-reference.md)
- [Python API](docs/python-api.md)
- [Architecture](docs/architecture.md)

## Requirements

- Python 3.10+
- libjpeg-turbo (for native DCT extraction; falls back to Pillow if unavailable)

## License

MIT
