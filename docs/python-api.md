# Python API

## Core

### DCTImage

The central data structure. Holds DCT coefficients for Y, Cb, Cr channels.

```python
from dct_vision.core.dct_image import DCTImage

# Load from JPEG (native libjpeg extraction, no pixel decode)
img = DCTImage.from_file("photo.jpg")

# Create from pixel array
import numpy as np
pixels = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
img = DCTImage.from_array(pixels, quality=85)

# Access coefficients
print(img.y_coeffs.shape)   # (32, 32, 8, 8) for 256x256
print(img.width, img.height) # 256, 256
print(img.num_components)    # 3

# Reconstruct pixels (for verification or display)
pixels = img.to_pixels()  # np.ndarray, uint8

# Save to JPEG (native lossless write when possible)
img.save("output.jpg")
```

**Attributes:**
- `y_coeffs` -- Luminance DCT coefficients, shape (bh, bw, 8, 8), dtype int16
- `cb_coeffs` -- Blue-difference chroma (None for grayscale)
- `cr_coeffs` -- Red-difference chroma (None for grayscale)
- `quant_tables` -- List of quantization tables, each (8, 8)
- `width`, `height` -- Original pixel dimensions
- `num_components` -- 1 (grayscale) or 3 (color)
- `comp_info` -- Per-component sampling factors and quant table index

## Operations

All operations return a new DCTImage. The original is never modified.

### Blur

```python
from dct_vision.ops.blur import blur

blurred = blur(img, sigma=2.0)
blurred = blur(img, sigma=3.0, channels="luma")  # blur only Y channel

# Cross-block strategy for large sigma (avoids block boundary seams)
blurred = blur(img, sigma=4.0, cross_block=True)
```

For `sigma > 2.0`, the `cross_block=True` flag is recommended. It uses a 3x3
block neighborhood to provide context across block boundaries, producing
smoother results at the cost of slightly more computation. See
[Architecture](architecture.md) for details on how the cross-block strategy works.

### Sharpen

```python
from dct_vision.ops.sharpen import sharpen

sharp = sharpen(img, amount=1.5)  # 1.0 = no change, 2.0 = strong
```

### Brightness

```python
from dct_vision.ops.color import adjust_brightness

bright = adjust_brightness(img, offset=30)   # positive = brighter
dark = adjust_brightness(img, offset=-20)
```

### Contrast

```python
from dct_vision.ops.color import adjust_contrast

high = adjust_contrast(img, factor=2.0)   # >1 = more contrast
low = adjust_contrast(img, factor=0.5)
```

### Downscale

```python
from dct_vision.ops.scale import downscale

half = downscale(img, factor=2)    # 256x256 -> 128x128
quarter = downscale(img, factor=4)
```

### Upscale (convenience wrapper)

```python
from dct_vision.ops.scale import upscale

big = upscale(img, factor=2)  # decodes to pixels, resizes via Pillow, re-encodes
```

### Edge Detection

```python
from dct_vision.ops.edge import detect_edges

edges_lap = detect_edges(img, method="laplacian")
edges_grad = detect_edges(img, method="gradient")
# Returns grayscale DCTImage
```

### Sobel / Scharr Edge Detection

```python
from dct_vision.ops.filters import sobel, scharr

edges = sobel(img, direction="both")        # or "horizontal", "vertical"
edges = scharr(img, direction="horizontal") # more accurate than Sobel
```

### Box Blur

```python
from dct_vision.ops.filters import box_blur

blurred = box_blur(img, radius=3)
```

### Emboss

```python
from dct_vision.ops.filters import emboss

relief = emboss(img, angle=45)   # angle in degrees
```

### Band-pass Filter

```python
from dct_vision.ops.filters import bandpass

# Keep only mid-frequency content (no OpenCV equivalent)
mid = bandpass(img, low_cutoff=2, high_cutoff=5)
```

### Unsharp Mask

```python
from dct_vision.ops.filters import unsharp_mask

sharp = unsharp_mask(img, sigma=2.0, amount=1.5)
```

### Quality Estimation

```python
from dct_vision.ops.quality import estimate_quality, dct_stats

q = estimate_quality(img)  # int, 1-100
stats = dct_stats(img)     # dict with dc_mean, ac_energy, etc.
```

## Augmentations

For ML data pipelines. All operate directly on DCT coefficients.

```python
from dct_vision.augment.flip import horizontal_flip, vertical_flip
from dct_vision.augment.crop import block_crop
from dct_vision.augment.jitter import brightness_jitter, contrast_jitter
from dct_vision.augment.noise import gaussian_noise

# Flip (pure coefficient sign manipulation)
flipped = horizontal_flip(img)
flipped = vertical_flip(img)

# Block-aligned crop (zero-copy slice)
cropped = block_crop(img, block_row=2, block_col=2, block_rows=8, block_cols=8)

# Random jitter (seeded for reproducibility)
jittered = brightness_jitter(img, max_offset=30, seed=42)
jittered = contrast_jitter(img, max_factor=0.3, seed=42)

# Gaussian noise on AC coefficients (DC preserved)
noisy = gaussian_noise(img, sigma=3.0, seed=42)
```

## Format Conversion

```python
from dct_vision.io.convert import convert_to_dct

# PNG, BMP, TIFF -> DCTImage
img = convert_to_dct("input.png", quality=85)
img.save("output.jpg")
```

## Operation Chaining

All operations return DCTImage, so they chain naturally:

```python
from dct_vision.core.dct_image import DCTImage
from dct_vision.ops.blur import blur
from dct_vision.ops.sharpen import sharpen
from dct_vision.ops.color import adjust_brightness

result = adjust_brightness(sharpen(blur(
    DCTImage.from_file("photo.jpg"),
    sigma=1.5,
), amount=1.3), offset=10)

result.save("processed.jpg")
```
