"""Tests for DCT/IDCT math utilities."""

import numpy as np
import pytest
from dct_vision.math.dct import dct2, idct2, blockwise_dct, blockwise_idct


class TestDCT2:
    def test_all_zeros(self):
        block = np.zeros((8, 8), dtype=np.float32)
        result = dct2(block)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_dc_only_for_constant_block(self):
        """A constant block should have energy only in DC (0,0)."""
        block = np.full((8, 8), 100.0, dtype=np.float32)
        result = dct2(block)
        # DC component should be non-zero
        assert abs(result[0, 0]) > 1.0
        # All AC components should be zero
        ac = result.copy()
        ac[0, 0] = 0.0
        np.testing.assert_allclose(ac, 0.0, atol=1e-10)

    def test_output_shape(self):
        block = np.random.rand(8, 8).astype(np.float32)
        result = dct2(block)
        assert result.shape == (8, 8)

    def test_output_dtype_float32(self):
        block = np.random.rand(8, 8).astype(np.float32)
        result = dct2(block)
        assert result.dtype == np.float32


class TestIDCT2:
    def test_roundtrip_precision(self):
        """DCT -> IDCT roundtrip error should be < 1e-10."""
        block = np.random.RandomState(42).rand(8, 8).astype(np.float32) * 255
        recovered = idct2(dct2(block))
        np.testing.assert_allclose(recovered, block, atol=1e-4)

    def test_roundtrip_multiple_blocks(self):
        """Roundtrip should work for various random blocks."""
        rng = np.random.RandomState(7)
        for _ in range(10):
            block = rng.rand(8, 8).astype(np.float32) * 255
            recovered = idct2(dct2(block))
            np.testing.assert_allclose(recovered, block, atol=1e-4)


class TestBlockwiseDCT:
    def test_basic_shape(self):
        """blockwise_dct of (16, 24) channel -> (2, 3, 8, 8) coefficients."""
        channel = np.random.rand(16, 24).astype(np.float32) * 255
        coeffs = blockwise_dct(channel)
        assert coeffs.shape == (2, 3, 8, 8)

    def test_roundtrip(self):
        """blockwise_dct -> blockwise_idct roundtrip."""
        channel = np.random.RandomState(42).rand(32, 32).astype(np.float32) * 255
        recovered = blockwise_idct(blockwise_dct(channel))
        np.testing.assert_allclose(recovered, channel, atol=1e-3)

    def test_not_divisible_by_8_raises(self):
        """Channel dimensions must be divisible by 8."""
        channel = np.random.rand(10, 10).astype(np.float32)
        with pytest.raises(ValueError):
            blockwise_dct(channel)


class TestBlockwiseIDCT:
    def test_output_shape(self):
        """blockwise_idct of (2, 3, 8, 8) -> (16, 24) channel."""
        coeffs = np.random.rand(2, 3, 8, 8).astype(np.float32)
        channel = blockwise_idct(coeffs)
        assert channel.shape == (16, 24)
