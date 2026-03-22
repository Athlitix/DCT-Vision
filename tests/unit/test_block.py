"""Tests for block utilities."""

import numpy as np
import pytest
from dct_vision.core.block import (
    pixel_to_block_coords,
    block_to_pixel_coords,
    pad_to_block_multiple,
    iter_blocks,
    BLOCK_SIZE,
)


class TestPixelToBlockCoords:
    def test_origin(self):
        assert pixel_to_block_coords(0, 0) == (0, 0)

    def test_within_first_block(self):
        assert pixel_to_block_coords(7, 7) == (0, 0)

    def test_second_block(self):
        assert pixel_to_block_coords(8, 0) == (1, 0)
        assert pixel_to_block_coords(0, 8) == (0, 1)

    def test_arbitrary(self):
        assert pixel_to_block_coords(20, 35) == (2, 4)


class TestBlockToPixelCoords:
    def test_origin(self):
        assert block_to_pixel_coords(0, 0) == (0, 0)

    def test_first_row(self):
        assert block_to_pixel_coords(0, 1) == (0, 8)

    def test_arbitrary(self):
        assert block_to_pixel_coords(3, 5) == (24, 40)


class TestPadToBlockMultiple:
    def test_already_multiple(self):
        channel = np.ones((16, 24), dtype=np.float32)
        padded = pad_to_block_multiple(channel)
        assert padded.shape == (16, 24)
        np.testing.assert_array_equal(padded, channel)

    def test_needs_padding(self):
        channel = np.ones((10, 13), dtype=np.float32)
        padded = pad_to_block_multiple(channel)
        assert padded.shape[0] % BLOCK_SIZE == 0
        assert padded.shape[1] % BLOCK_SIZE == 0
        assert padded.shape == (16, 16)

    def test_padding_is_zeros(self):
        channel = np.ones((10, 10), dtype=np.float32)
        padded = pad_to_block_multiple(channel)
        # Original region preserved
        np.testing.assert_array_equal(padded[:10, :10], 1.0)
        # Padding region is zero
        np.testing.assert_array_equal(padded[10:, :], 0.0)
        np.testing.assert_array_equal(padded[:, 10:], 0.0)

    def test_single_pixel(self):
        channel = np.array([[42.0]], dtype=np.float32)
        padded = pad_to_block_multiple(channel)
        assert padded.shape == (8, 8)
        assert padded[0, 0] == 42.0


class TestIterBlocks:
    def test_basic(self):
        channel = np.arange(256, dtype=np.float32).reshape(16, 16)
        blocks = list(iter_blocks(channel))
        assert len(blocks) == 4  # 2x2 blocks
        # Each block is (row, col, 8x8 array)
        row, col, block = blocks[0]
        assert row == 0 and col == 0
        assert block.shape == (8, 8)

    def test_block_content(self):
        channel = np.arange(256, dtype=np.float32).reshape(16, 16)
        blocks = list(iter_blocks(channel))
        _, _, first_block = blocks[0]
        expected = channel[:8, :8]
        np.testing.assert_array_equal(first_block, expected)

    def test_non_multiple_raises(self):
        channel = np.ones((10, 10), dtype=np.float32)
        with pytest.raises(ValueError):
            list(iter_blocks(channel))
