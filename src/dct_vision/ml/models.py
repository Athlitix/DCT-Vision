"""CNN architectures for DCT and pixel input classification.

Three models for comparison:
1. PixelCNN -- standard RGB pixel input baseline
2. DCTVanillaCNN -- same architecture, 64-channel DCT input
3. DCTFreqBranchCNN -- separate low/mid/high frequency branches (Uber-inspired)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PixelCNN(nn.Module):
    """Baseline CNN for pixel (RGB) input.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_channels : int
        Number of input channels (3 for RGB).
    input_size : int
        Spatial input size (32 for CIFAR, 96 for STL-10).
    """

    def __init__(self, num_classes: int = 10, input_channels: int = 3, input_size: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class DCTVanillaCNN(nn.Module):
    """Vanilla CNN with DCT coefficient input.

    Same architecture as PixelCNN but accepts 64-channel (or 192-channel)
    DCT coefficient tensors instead of 3-channel RGB.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_channels : int
        Number of input channels (64 for y_only, 192 for ycbcr).
    block_grid : tuple[int, int]
        Spatial dimensions in blocks (4,4 for CIFAR, 12,12 for STL-10).
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 64,
        block_grid: tuple[int, int] = (4, 4),
    ):
        super().__init__()
        # 1x1 conv to reduce DCT channels to manageable size
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(input_channels, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.features = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2) if min(block_grid) >= 4 else nn.Identity(),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_reduce(x)
        x = self.features(x)
        return self.classifier(x)


class DCTFreqBranchCNN(nn.Module):
    """Frequency-branch CNN for DCT input (Uber 2018 inspired).

    Splits 64 DCT channels into three frequency bands:
    - Low (DC + first few AC): channels 0-15 (frequencies 0-1 in u,v)
    - Mid (mid-range AC): channels 16-47 (frequencies 2-5)
    - High (highest AC): channels 48-63 (frequencies 6-7)

    Each branch processes its band independently, then features are
    merged for classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    block_grid : tuple[int, int]
        Spatial dimensions in blocks.
    """

    def __init__(
        self,
        num_classes: int = 10,
        block_grid: tuple[int, int] = (4, 4),
    ):
        super().__init__()

        # Frequency band channel indices
        # DCT coefficients are laid out as 8x8 -> 64 channels
        # Low frequencies: positions where max(u,v) <= 1 -> indices for (0,0),(0,1),(1,0),(1,1)
        # We split by zigzag-like ordering: first 16 = low, next 32 = mid, last 16 = high
        self._low_channels = 16
        self._mid_channels = 32
        self._high_channels = 16

        use_pool = min(block_grid) >= 4

        def _make_branch(in_ch, out_ch):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ]
            if use_pool:
                layers.append(nn.MaxPool2d(2))
            layers.extend([
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            ])
            return nn.Sequential(*layers)

        self.low_branch = _make_branch(self._low_channels, 64)
        self.mid_branch = _make_branch(self._mid_channels, 64)
        self.high_branch = _make_branch(self._high_channels, 64)

        # Merge
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.low_branch(x[:, :self._low_channels])
        mid = self.mid_branch(x[:, self._low_channels:self._low_channels + self._mid_channels])
        high = self.high_branch(x[:, self._low_channels + self._mid_channels:])
        merged = torch.cat([low, mid, high], dim=1)
        return self.classifier(merged)


class PixelResNet18(nn.Module):
    """ResNet-18 adapted for small pixel images (CIFAR-style).

    Standard torchvision ResNet-18 with a 3x3 stride-1 stem and no max-pool,
    the usual CIFAR adaptation (the 7x7 stride-2 stem + maxpool would destroy a
    32x32 image). Serves as the RGB baseline for the DCT ResNet comparison.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_channels : int
        Number of input channels (3 for RGB).
    """

    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super().__init__()
        from torchvision.models import resnet18

        net = resnet18(weights=None, num_classes=num_classes)
        net.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DCTResNet18(nn.Module):
    """ResNet-18 with a DCT-coefficient stem (Uber 2018, "Faster Neural
    Networks Straight from JPEG").

    DCT input is already 8x spatially downsampled (one vector per 8x8 block), so
    the standard stride-2 stem + maxpool is replaced by a stride-1 stem that
    maps the 64- or 192-channel block tensor into ResNet's 64-channel trunk.
    For a CIFAR 4x4 block grid we also drop the first residual stage's stride so
    spatial resolution is not collapsed to 1x1 before the network can use it.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_channels : int
        DCT channels (64 for y_only, 192 for ycbcr).
    block_grid : tuple[int, int]
        Spatial size in blocks (used to decide whether to keep early strides).
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 64,
        block_grid: tuple[int, int] = (4, 4),
    ):
        super().__init__()
        from torchvision.models import resnet18

        net = resnet18(weights=None, num_classes=num_classes)
        # Stem: 1x1 conv mapping DCT channels -> 64, stride 1, no maxpool.
        net.conv1 = nn.Conv2d(input_channels, 64, kernel_size=1, stride=1, padding=0, bias=False)
        net.maxpool = nn.Identity()

        # For tiny block grids, relax early downsampling so we don't hit 1x1
        # before layer4. A 4x4 grid -> layer2/3/4 would give 2x2,1x1,1x1.
        if min(block_grid) <= 4:
            _set_stage_stride(net.layer2, 1)

        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _set_stage_stride(stage: nn.Module, stride: int) -> None:
    """Set the stride of a ResNet stage's first block (conv + downsample)."""
    block = stage[0]
    if hasattr(block, "conv1") and block.conv1.stride != (stride, stride):
        block.conv1.stride = (stride, stride)
    if getattr(block, "downsample", None) is not None:
        block.downsample[0].stride = (stride, stride)
