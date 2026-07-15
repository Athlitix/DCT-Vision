"""Tests for ML model architectures."""

import pytest
import numpy as np

torch = pytest.importorskip("torch")

from dct_vision.ml.models import (
    PixelCNN,
    DCTVanillaCNN,
    DCTFreqBranchCNN,
    PixelResNet18,
    DCTResNet18,
)


class TestPixelCNN:
    def test_forward_shape(self):
        model = PixelCNN(num_classes=10, input_channels=3)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10)

    def test_stl10_input(self):
        model = PixelCNN(num_classes=10, input_channels=3, input_size=96)
        x = torch.randn(2, 3, 96, 96)
        out = model(x)
        assert out.shape == (2, 10)


class TestDCTVanillaCNN:
    def test_forward_cifar(self):
        """CIFAR-10: 32x32 pixels = 4x4 blocks, 64 channels."""
        model = DCTVanillaCNN(num_classes=10, input_channels=64, block_grid=(4, 4))
        x = torch.randn(4, 64, 4, 4)
        out = model(x)
        assert out.shape == (4, 10)

    def test_forward_stl10(self):
        """STL-10: 96x96 pixels = 12x12 blocks, 64 channels."""
        model = DCTVanillaCNN(num_classes=10, input_channels=64, block_grid=(12, 12))
        x = torch.randn(2, 64, 12, 12)
        out = model(x)
        assert out.shape == (2, 10)

    def test_ycbcr_mode(self):
        """192 channels (Y+Cb+Cr)."""
        model = DCTVanillaCNN(num_classes=10, input_channels=192, block_grid=(4, 4))
        x = torch.randn(4, 192, 4, 4)
        out = model(x)
        assert out.shape == (4, 10)

    def test_parameter_count_reasonable(self):
        model = DCTVanillaCNN(num_classes=10, input_channels=64, block_grid=(4, 4))
        params = sum(p.numel() for p in model.parameters())
        assert params < 5_000_000  # reasonable for a small model


class TestDCTFreqBranchCNN:
    def test_forward_cifar(self):
        model = DCTFreqBranchCNN(num_classes=10, block_grid=(4, 4))
        x = torch.randn(4, 64, 4, 4)
        out = model(x)
        assert out.shape == (4, 10)

    def test_forward_stl10(self):
        model = DCTFreqBranchCNN(num_classes=10, block_grid=(12, 12))
        x = torch.randn(2, 64, 12, 12)
        out = model(x)
        assert out.shape == (2, 10)

    def test_has_three_branches(self):
        model = DCTFreqBranchCNN(num_classes=10, block_grid=(4, 4))
        assert hasattr(model, 'low_branch')
        assert hasattr(model, 'mid_branch')
        assert hasattr(model, 'high_branch')

    def test_parameter_count_reasonable(self):
        model = DCTFreqBranchCNN(num_classes=10, block_grid=(4, 4))
        params = sum(p.numel() for p in model.parameters())
        assert params < 5_000_000


class TestResNet18:
    def test_pixel_resnet_cifar(self):
        model = PixelResNet18(num_classes=10, input_channels=3)
        out = model(torch.randn(4, 3, 32, 32))
        assert out.shape == (4, 10)

    def test_dct_resnet_cifar_y_only(self):
        model = DCTResNet18(num_classes=10, input_channels=64, block_grid=(4, 4))
        out = model(torch.randn(4, 64, 4, 4))
        assert out.shape == (4, 10)

    def test_dct_resnet_cifar_ycbcr(self):
        model = DCTResNet18(num_classes=10, input_channels=192, block_grid=(4, 4))
        out = model(torch.randn(4, 192, 4, 4))
        assert out.shape == (4, 10)

    def test_dct_resnet_stl10(self):
        model = DCTResNet18(num_classes=10, input_channels=64, block_grid=(12, 12))
        out = model(torch.randn(2, 64, 12, 12))
        assert out.shape == (2, 10)

    def test_dct_resnet_train_step(self):
        model = DCTResNet18(num_classes=10, input_channels=64, block_grid=(4, 4))
        x = torch.randn(4, 64, 4, 4)
        y = torch.randint(0, 10, (4,))
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()


class TestModelTrainStep:
    """Verify models can do a forward + backward pass."""

    def test_pixel_train_step(self):
        model = PixelCNN(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 10, (4,))
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()

    def test_dct_vanilla_train_step(self):
        model = DCTVanillaCNN(num_classes=10, input_channels=64, block_grid=(4, 4))
        x = torch.randn(4, 64, 4, 4)
        y = torch.randint(0, 10, (4,))
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()

    def test_dct_freq_branch_train_step(self):
        model = DCTFreqBranchCNN(num_classes=10, block_grid=(4, 4))
        x = torch.randn(4, 64, 4, 4)
        y = torch.randint(0, 10, (4,))
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
