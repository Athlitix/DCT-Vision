"""Tests for JPEG DCT coefficient extraction and writing."""

import numpy as np
import pytest
from pathlib import Path

from dct_vision._libjpeg.bindings import read_dct_coefficients, write_dct_coefficients
from dct_vision.exceptions import InvalidImageError


class TestReadDCTCoefficients:
    def test_returns_dict_with_expected_keys(self, sample_jpeg):
        result = read_dct_coefficients(str(sample_jpeg))
        assert "coefficients" in result
        assert "quant_tables" in result
        assert "width" in result
        assert "height" in result
        assert "num_components" in result

    def test_coefficient_shapes(self, sample_jpeg):
        """256x256 JPEG should have 32x32 blocks for luma."""
        result = read_dct_coefficients(str(sample_jpeg))
        coeffs = result["coefficients"]
        assert len(coeffs) >= 1
        y_coeffs = coeffs[0]
        assert y_coeffs.ndim == 4
        assert y_coeffs.shape[2] == 8
        assert y_coeffs.shape[3] == 8

    def test_quant_tables_shape(self, sample_jpeg):
        result = read_dct_coefficients(str(sample_jpeg))
        qtables = result["quant_tables"]
        assert len(qtables) >= 1
        assert qtables[0].shape == (8, 8)

    def test_dimensions(self, sample_jpeg):
        result = read_dct_coefficients(str(sample_jpeg))
        assert result["width"] == 256
        assert result["height"] == 256

    def test_color_image_has_3_components(self, sample_jpeg):
        result = read_dct_coefficients(str(sample_jpeg))
        assert result["num_components"] == 3
        assert len(result["coefficients"]) == 3

    def test_grayscale_has_1_component(self, grayscale_jpeg):
        result = read_dct_coefficients(str(grayscale_jpeg))
        assert result["num_components"] == 1
        assert len(result["coefficients"]) == 1

    def test_single_block_image(self, single_block_jpeg):
        result = read_dct_coefficients(str(single_block_jpeg))
        y_coeffs = result["coefficients"][0]
        assert y_coeffs.shape[0] == 1
        assert y_coeffs.shape[1] == 1

    def test_420_subsampling_chroma_smaller(self, sub_420_jpeg):
        result = read_dct_coefficients(str(sub_420_jpeg))
        y = result["coefficients"][0]
        cb = result["coefficients"][1]
        # For 4:2:0, chroma blocks should be fewer than luma
        assert cb.shape[0] <= y.shape[0]
        assert cb.shape[1] <= y.shape[1]

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(InvalidImageError):
            read_dct_coefficients(str(tmp_path / "nonexistent.jpg"))

    def test_non_jpeg_raises(self, sample_png):
        with pytest.raises(InvalidImageError):
            read_dct_coefficients(str(sample_png))

    def test_coefficients_dtype_int16(self, sample_jpeg):
        result = read_dct_coefficients(str(sample_jpeg))
        y_coeffs = result["coefficients"][0]
        assert y_coeffs.dtype == np.int16


class TestWriteDCTCoefficients:
    def test_write_produces_file(self, sample_jpeg, tmp_path):
        result = read_dct_coefficients(str(sample_jpeg))
        out_path = tmp_path / "output.jpg"
        write_dct_coefficients(
            str(out_path),
            result["coefficients"],
            result["quant_tables"],
            result["width"],
            result["height"],
            result["num_components"],
            result.get("comp_info"),
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_output_is_valid_jpeg(self, sample_jpeg, tmp_path):
        result = read_dct_coefficients(str(sample_jpeg))
        out_path = tmp_path / "valid.jpg"
        write_dct_coefficients(
            str(out_path),
            result["coefficients"],
            result["quant_tables"],
            result["width"],
            result["height"],
            result["num_components"],
            result.get("comp_info"),
        )
        with open(out_path, "rb") as f:
            magic = f.read(2)
        assert magic == b"\xff\xd8"

    def test_roundtrip_preserves_dimensions(self, sample_jpeg, tmp_path):
        """Read -> write -> read should preserve image dimensions."""
        result = read_dct_coefficients(str(sample_jpeg))
        out_path = tmp_path / "roundtrip.jpg"
        write_dct_coefficients(
            str(out_path),
            result["coefficients"],
            result["quant_tables"],
            result["width"],
            result["height"],
            result["num_components"],
            result.get("comp_info"),
        )
        result2 = read_dct_coefficients(str(out_path))
        assert result2["width"] == result["width"]
        assert result2["height"] == result["height"]
        assert result2["num_components"] == result["num_components"]

    def test_roundtrip_visual_similarity(self, sample_jpeg, tmp_path):
        """Read -> write -> visual comparison should be close."""
        from PIL import Image
        original = np.array(Image.open(str(sample_jpeg)), dtype=np.float32)

        result = read_dct_coefficients(str(sample_jpeg))
        out_path = tmp_path / "roundtrip.jpg"
        write_dct_coefficients(
            str(out_path),
            result["coefficients"],
            result["quant_tables"],
            result["width"],
            result["height"],
            result["num_components"],
            result.get("comp_info"),
        )
        reconstructed = np.array(Image.open(str(out_path)), dtype=np.float32)

        # PSNR should be reasonable (> 25dB) — not exact due to re-encode
        mse = np.mean((original - reconstructed) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(255.0**2 / mse)
            assert psnr > 25.0, f"PSNR too low: {psnr:.1f}dB"

    def test_grayscale_roundtrip(self, grayscale_jpeg, tmp_path):
        result = read_dct_coefficients(str(grayscale_jpeg))
        out_path = tmp_path / "gray_roundtrip.jpg"
        write_dct_coefficients(
            str(out_path),
            result["coefficients"],
            result["quant_tables"],
            result["width"],
            result["height"],
            result["num_components"],
            result.get("comp_info"),
        )
        assert out_path.exists()
        result2 = read_dct_coefficients(str(out_path))
        assert result2["num_components"] == 1
