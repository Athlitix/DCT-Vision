# DCT-Vision Benchmark Results

Honest, reproducible measurements of frequency-domain image processing against
Pillow and OpenCV, plus the ML validation experiments. All numbers are produced
by the scripts in `benchmarks/` and the training harness in
`dct_vision.ml.train`.

## Methodology

- **Timing:** `time.perf_counter_ns`, 5 warmup iterations discarded, 100 timed
  repetitions, mean reported. Single machine, RTX 3060 Laptop / 20-core CPU.
- **Op-only vs end-to-end:** op-only times the operation with the image already
  in memory. End-to-end times the full pipeline (load + op + save): DCT writes
  modified coefficients back losslessly (`save(quality=None)`); the pixel
  pipeline does decode + op + encode. End-to-end is the fairer headline because
  the entire premise is skipping the decode.
- **Quality:** PSNR/SSIM of the DCT-domain op vs the *same op applied in the
  spatial domain to our own decoded pixels*, so the metric isolates operation
  error and does not conflate decoder or JPEG-roundtrip differences. Lossless
  transforms (rotation, flips, transpose) are additionally proven exact in the
  unit tests (PSNR > 55 dB vs `numpy` ground truth).
- **Coverage:** resolutions 256/512/1024/2048, JPEG qualities 50/75/85/95.

Reproduce:

```bash
uv run python benchmarks/run_all.py -o benchmarks/results/latest.json
uv run python benchmarks/cross_block_sweep.py -o benchmarks/results/cross_block.json
uv run python -m dct_vision.ml.train --models pixelcnn,dctcnn,dctfreq \
    --epochs 15 -o benchmarks/results/ml2_cifar.json
uv run python -m dct_vision.ml.train --models pixelresnet,dctresnet --augment \
    --epochs 20 -o benchmarks/results/ml3_resnet_aug.json
```

## Classical operations

See `latest.json` for the full per-resolution/quality table. Headline reading:

- **Large op-only wins vs Pillow** (blur, sharpen, brightness, contrast, edges):
  roughly 12-20x, because these are pointwise coefficient operations.
- **vs OpenCV is nuanced and honest:** brightness/contrast/noise beat OpenCV
  several-fold, blur is about par, but OpenCV's SIMD wins on flips, rotate,
  crop, downscale and edges. DCT-Vision does not claim to beat OpenCV on
  everything.
- **End-to-end** favours DCT on expensive ops (contrast ~1.5x, noise ~1.9x) and
  large images, and is roughly par on cheap ops (libjpeg-turbo decode is fast,
  so avoiding it saves less on a 0.1 ms operation).

### Lossless transforms

`rotate90/180/270` and `transpose` are exact coefficient permutations
(jpegtran-style): no IDCT/DCT and zero quality loss, proven in the unit tests.

### Cross-block blur (seam analysis)

`cross_block.json`: the overlapping-block strategy removes 8x8 seams
(blockiness ~0 vs 7-14 for per-block; PSNR up to 42 dB vs ~15 dB at sigma = 8)
but costs ~1000x more (~200 ms vs ~0.2 ms at 512px) because it performs
neighborhood IDCT/DCT. Worth it only at large sigma, where a pixel pipeline
would decode anyway.

## ML experiments

DCT coefficients are pre-computed to an in-memory tensor cache once (ML-1c), so
the data pipeline does no per-epoch DCT work.

### ML-2: does a DCT-input CNN train correctly? (CIFAR-10, 15 epochs, no aug)

| Model     | Input       | Params  | Test acc | Train time | Data time |
|-----------|-------------|---------|----------|------------|-----------|
| PixelCNN  | RGB pixels  | 590 794 | 0.833    | 117.2 s    | 6.1 s     |
| DCTVanillaCNN | DCT y_only | 556 106 | 0.579 | 16.8 s     | 3.9 s     |
| DCTFreqBranchCNN | DCT y_only | 174 602 | 0.520 | 19.4 s | 4.0 s     |

### ML-3/ML-4: ResNet-18, pixel aug vs DCT aug (CIFAR-10, 20 epochs)

| Model        | Input      | Params     | Test acc | Train time | Data time |
|--------------|------------|------------|----------|------------|-----------|
| PixelResNet18 | RGB pixels + hflip/crop | 11.17 M | 0.896 | 723.6 s | 10.6 s |
| DCTResNet18  | DCT y_only + DCT hflip  | 11.18 M | 0.638 | 88.1 s  | 7.2 s  |

### Honest interpretation

DCT input yields a **large, consistent training-speed advantage** (~7x for the
CNN, ~8x for ResNet-18) because the network sees a 4x4x64 tensor instead of a
32x32x3 image — far less spatial compute.

The **accuracy gap does not close** on CIFAR-10 in this setup. This is expected,
not a failure of the representation: CIFAR images are only 32x32, so DCT's 8x
block downsampling leaves a 4x4 spatial grid, discarding spatial detail these
architectures rely on. The Uber 2018 accuracy-parity result was on ImageNet,
where 8x downsampling still leaves ~28x28 blocks. The path to parity here is
larger inputs (STL-10, ImageNet), the 192-channel `ycbcr` mode, and
architectures designed for the DCT block grid — future work, honestly labelled.
