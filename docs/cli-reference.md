# CLI Reference

After installation, the `dv` command is available globally.

## Global flags

```
--version        Show version and exit
--verbose / -v   Increase log level (stackable: -vv for DEBUG)
--quiet / -q     Suppress all output except errors
--help           Show help
```

## Image Operations

### blur

Apply Gaussian blur in DCT domain.

```bash
dv blur input.jpg -o output.jpg --sigma 2.0
dv blur input.jpg -o output.jpg --sigma 3.0 --channels luma
dv blur input.jpg -o output.jpg --sigma 4.0 --cross-block
dv blur input.jpg -o output.jpg --sigma 2.0 --timing
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o / --output` | required | Output file path |
| `-s / --sigma` | 2.0 | Gaussian sigma (higher = more blur) |
| `--channels` | all | Channels to process: all, luma, chroma |
| `--cross-block` | false | Use 3x3 block neighborhood for cross-boundary smoothness |
| `-t / --timing` | false | Print execution time |
| `--memory` | false | Print peak memory usage |

The `--cross-block` flag is recommended for `sigma > 2.0`. Without it, block
boundary seams may be visible at high sigma values.

### sharpen

Sharpen by boosting high-frequency DCT coefficients.

```bash
dv sharpen input.jpg -o output.jpg --amount 1.5
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o / --output` | required | Output file path |
| `-a / --amount` | 1.5 | Sharpening strength (1.0 = no change) |
| `-t / --timing` | false | Print execution time |

### brightness

Adjust brightness by modifying DC coefficients.

```bash
dv brightness input.jpg -o output.jpg --offset 30
dv brightness input.jpg -o output.jpg --offset -20
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o / --output` | required | Output file path |
| `--offset` | 30.0 | Brightness offset (positive = brighter) |
| `-t / --timing` | false | Print execution time |

### contrast

Adjust contrast by scaling AC coefficients.

```bash
dv contrast input.jpg -o output.jpg --factor 1.5
dv contrast input.jpg -o output.jpg --factor 0.5
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o / --output` | required | Output file path |
| `-f / --factor` | 1.5 | Contrast factor (>1 = more contrast) |
| `-t / --timing` | false | Print execution time |

### downscale

Downscale via DCT-domain block merging.

```bash
dv downscale input.jpg -o output.jpg --factor 2
dv downscale input.jpg -o output.jpg --factor 4
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o / --output` | required | Output file path |
| `--factor` | 2 | Downscale factor (must be power of 2) |
| `-t / --timing` | false | Print execution time |

### edges

Detect edges in DCT domain.

```bash
dv edges input.jpg -o edges.jpg --method laplacian
dv edges input.jpg -o edges.jpg --method gradient
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o / --output` | required | Output file path |
| `-m / --method` | laplacian | Edge method: laplacian or gradient |
| `-t / --timing` | false | Print execution time |

### convert

Convert non-JPEG formats to JPEG via DCT representation.

```bash
dv convert input.png -o output.jpg --quality 85
dv convert input.bmp -o output.jpg --quality 95
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o / --output` | required | Output file path |
| `-q / --quality` | 85 | JPEG quality (1-100) |
| `-t / --timing` | false | Print execution time |

## Inspection / Analysis

### info

Display DCT image information.

```bash
dv info input.jpg
dv info input.jpg --json
```

Output includes: dimensions, component count, subsampling mode, estimated quality, DC statistics, AC energy, nonzero coefficient ratio.

### inspect

Dump DCT coefficients for a specific 8x8 block.

```bash
dv inspect input.jpg --block 0,0
dv inspect input.jpg --block 5,10 --json
```

### quality

Estimate JPEG quality factor from quantization tables.

```bash
dv quality input.jpg
```

## Augmentation (ML pipelines)

### augment

Apply DCT-domain augmentations. All augmentations happen directly on coefficients without pixel decode.

```bash
dv augment input.jpg -o output.jpg --flip horizontal
dv augment input.jpg -o output.jpg --flip vertical --brightness-jitter 20 --noise 3.0 --seed 42
dv augment input.jpg -o output.jpg --contrast-jitter 0.3 --seed 99
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o / --output` | required | Output file path |
| `--flip` | none | Flip direction: horizontal or vertical |
| `--brightness-jitter` | 0.0 | Max random brightness offset |
| `--contrast-jitter` | 0.0 | Max random contrast deviation |
| `--noise` | 0.0 | Gaussian noise sigma |
| `--seed` | none | Random seed for reproducibility |
| `-t / --timing` | false | Print execution time |
