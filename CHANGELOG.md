# Changelog

All notable changes to DCT-Vision are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow semver.

## [0.3.0] - 2026-07-14

### Added
- `dct_vision.math.metrics`: PSNR, SSIM (Wang 2004, validated against
  scikit-image), and MSE -- pure numpy/scipy, shipped in production.
- Lossless geometric transforms `dct_vision.ops.geometry`: `rotate90/180/270`,
  `transpose`, and `rotate(degrees)`; bit-exact on all chroma subsampling modes.
  New CLI: `dv rotate`.
- ML models: `PixelResNet18` and `DCTResNet18` (Uber-2018 style DCT stem).
- Applications `dct_vision.apps` and `dv apps` CLI: near-duplicate detection
  (`find_duplicates`), JPEG double-compression forensics
  (`detect_double_compression`), and DC-only thumbnails (`dc_thumbnail`).
- Benchmarks now record PSNR/SSIM quality, peak-memory ratio, and end-to-end
  (load+op+save) throughput. New `benchmarks/cross_block_sweep.py` seam analysis.
- Reproducible, seeded augmentation with correct per-DataLoader-worker streams;
  `AugmentationPipeline(seed=...)` and `DCTDataset(seed=...)`. Rotation
  augmentations (`rot90/rot180/rot270`).
- ML training harness reworked (`dct_vision.ml.train`): in-memory DCT precompute
  cache, model selection, tensor-level DCT hflip augmentation, JSON output.
  Committed results under `benchmarks/results/` (ML-2, ML-3/4, YCbCr).
- `benchmarks/results/RESULTS.md` with methodology and honest findings.

### Fixed
- Brightness DC-offset was 8x too small (DC = mean*8 for orthonormal DCT); a
  brightness offset now shifts pixels by the requested amount.
- Grayscale JPEG writer omitted the +128 level shift, corrupting saved grayscale
  images from `from_array` (~7.7 dB -> ~50 dB).
- Transpose/rotation now transpose the (asymmetric) quantization tables too, so
  subsampled JPEGs are bit-exact (~25 dB -> ~99 dB).
- Downscale ~5x faster at 1024px+ (BLAS matmul instead of off-BLAS einsum),
  variance eliminated.

### Notes
- ML finding: DCT input trains ~7-8x faster but does not match pixel accuracy on
  CIFAR-10; YCbCr recovers 4-7 points over Y-only. See RESULTS.md.

## [0.2.0] - 2026-03-22

- 34 classical DCT-domain operations, cross-block strategy, native libjpeg cffi
  bindings, 3-way benchmarking (DCT vs Pillow vs OpenCV), DCT-domain
  augmentations, PyTorch `DCTDataset` + caching, packaging and CI. Published to
  PyPI.

## [0.1.0] - 2026-03-22

- Initial release: DCT extraction pipeline, losslessness verification, core
  `DCTImage`, JPEG I/O, and format converters.

[0.3.0]: https://github.com/Athlitix/DCT-Vision/releases/tag/v0.3.0
[0.2.0]: https://github.com/Athlitix/DCT-Vision/releases/tag/v0.2.0
[0.1.0]: https://github.com/Athlitix/DCT-Vision/releases/tag/v0.1.0
