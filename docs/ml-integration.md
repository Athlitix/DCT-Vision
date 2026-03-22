# ML Integration

## PyTorch DCTDataset

Load JPEG images as DCT coefficient tensors for ML training.
No pixel decode in the data pipeline.

### Basic usage

```python
from dct_vision.ml.dataset import DCTDataset
from torch.utils.data import DataLoader

# ImageFolder-style: train/cat/001.jpg, train/dog/002.jpg
dataset = DCTDataset(
    root="train/",
    mode="y_only",           # 64 channels per block position
    resize_blocks=(4, 4),    # fixed output: (64, 4, 4) tensor
)

loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for batch_tensor, batch_labels in loader:
    # batch_tensor: (32, 64, 4, 4)
    # batch_labels: (32,)
    output = model(batch_tensor)
```

### Tensor modes

| Mode | Shape | Description |
|------|-------|-------------|
| `y_only` | `(64, bh, bw)` | Luma channel, 64 coefficients per block |
| `ycbcr` | `(192, bh, bw)` | Y(64) + Cb(64) + Cr(64) channels |
| `dc_only` | `(3, bh, bw)` | Just DC values (fast, low-res features) |

### With augmentations

```python
dataset = DCTDataset(
    root="train/",
    mode="y_only",
    resize_blocks=(4, 4),
    augmentations=[
        "hflip:p=0.5",
        "brightness_jitter:max_offset=20",
        "noise:sigma=2.0",
    ],
)
```

All augmentations run in DCT domain -- no pixel decode.

### Augmentation pipeline

```python
from dct_vision.ml.augment_pipeline import AugmentationPipeline

# String format
pipe = AugmentationPipeline(["hflip:p=0.5", "noise:sigma=3.0"])

# Dict format
pipe = AugmentationPipeline([
    {"name": "hflip", "p": 0.5},
    {"name": "brightness_jitter", "max_offset": 20},
    {"name": "crop", "block_rows": 4, "block_cols": 4},
])
```

Supported augmentations: hflip, vflip, brightness_jitter, contrast_jitter, noise, crop.

## Dataset caching

Pre-extract coefficients to .npz for instant loading on subsequent epochs.

### CLI

```bash
# Pre-extract DCT coefficients
dv dataset prepare ./train/ -o ./train_dct/

# Show dataset stats
dv dataset info ./train/

# Benchmark loading speed
dv dataset bench ./train/ --mode y_only --batch-size 32 --resize 4,4
```

### Python

```python
from dct_vision.ml.cache import prepare_cache, load_cached, dataset_info

# Pre-extract
stats = prepare_cache("train/", "train_dct/")
print(f"Cached {stats['count']} images")

# Load cached
img = load_cached("train_dct/cat/001.npz")

# Stats
info = dataset_info("train/")
```
