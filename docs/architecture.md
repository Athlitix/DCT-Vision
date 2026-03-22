# Architecture

## How it works

JPEG images store pixel data as Discrete Cosine Transform (DCT) coefficients. Traditional image processing decodes these back to pixels, processes them, then re-encodes. DCT-Vision skips the decode/encode entirely -- it operates directly on the DCT coefficients.

```
Traditional:  JPEG -> decode -> pixels -> process -> pixels -> encode -> JPEG
DCT-Vision:   JPEG -> read coefficients -> process coefficients -> write coefficients -> JPEG
```

## Project structure

```
src/dct_vision/
    __init__.py              # Public API, version
    exceptions.py            # DCTVisionError hierarchy
    core/
        dct_image.py         # DCTImage -- central data structure
        block.py             # 8x8 block iteration, padding
        channel.py           # Y/Cb/Cr handling, subsampling
    io/
        jpeg_reader.py       # High-level JPEG reader
        jpeg_writer.py       # High-level JPEG writer
        convert.py           # PNG/BMP/TIFF -> DCTImage
        validation.py        # File format detection, input validation
    ops/
        blur.py              # Gaussian blur (frequency envelope)
        sharpen.py           # Sharpening (high-freq boost)
        color.py             # Brightness (DC offset), contrast (AC scale)
        scale.py             # Downscale (block merging), upscale (decode fallback)
        edge.py              # Edge detection (Laplacian, gradient)
        quality.py           # Quality estimation, DCT statistics
    augment/
        flip.py              # Horizontal/vertical flip
        crop.py              # Block-aligned crop
        jitter.py            # Brightness/contrast jitter
        noise.py             # Frequency-domain noise injection
    math/
        dct.py               # DCT/IDCT wrappers (scipy-backed)
        quantization.py      # Quantize/dequantize, quality scaling
        colorspace.py        # RGB <-> YCbCr conversion
    cli/
        app.py               # Typer CLI (dv command)
    utils/
        constants.py         # JPEG constants, standard quant tables
        logging.py           # Logging configuration
        profiling.py         # Timing, memory measurement
    _libjpeg/
        native.py            # cffi bindings for direct DCT coefficient access
        bindings.py          # Pillow+scipy fallback path
```

## I/O paths

### Native path (fast)

When `libjpeg-turbo` dev headers are available, DCT-Vision uses cffi bindings to call `jpeg_read_coefficients()` and `jpeg_write_coefficients()` directly. This extracts/writes DCT coefficients without any pixel decode/encode.

### Fallback path

Without native libjpeg access, the library decodes via Pillow, converts to YCbCr, computes blockwise DCT with scipy, and quantizes. This is slower but always works.

The path is selected automatically -- no configuration needed.

## Why operations are fast in DCT domain

| Operation | Spatial domain | DCT domain |
|-----------|---------------|------------|
| Blur | Convolve with Gaussian kernel (O(n*k)) | Multiply coefficients by envelope (O(n)) |
| Brightness | Add offset to every pixel | Add offset to DC coefficient only |
| Contrast | Scale every pixel relative to mean | Scale AC coefficients only |
| Sharpen | Unsharp mask (blur + subtract) | Multiply high-freq coefficients |
| Edge detect | Convolve with Laplacian kernel | Multiply by frequency weights |

The mathematical basis is the convolution theorem: convolution in spatial domain equals pointwise multiplication in frequency domain.

## Dependencies

Production (shipped with package):
- `numpy` -- array operations
- `scipy` -- DCT/IDCT computation
- `Pillow` -- non-JPEG format reading, fallback I/O
- `typer` + `rich` -- CLI
- `cffi` -- libjpeg-turbo bindings

Dev only:
- `pytest`, `pytest-cov` -- testing
- `opencv-python-headless` -- benchmark comparison
