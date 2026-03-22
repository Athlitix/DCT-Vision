"""Integration tests for DCT extraction and reconstruction pipeline.

Validates that our pipeline produces visually correct results
and that roundtrip operations preserve image quality.
"""

import numpy as np
import pytest
from PIL import Image
from pathlib import Path

from dct_vision.core.dct_image import DCTImage
from dct_vision.io.jpeg_reader import read_jpeg
from dct_vision.io.jpeg_writer import write_jpeg
from dct_vision.io.convert import convert_to_dct


def _psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)


class TestJPEGRoundtrip:
    """Load JPEG -> DCTImage -> to_pixels -> compare with Pillow decode."""

    def test_color_image_visual_quality(self, sample_jpeg):
        """Reconstructed pixels should be visually close to Pillow's decode."""
        original = np.array(Image.open(str(sample_jpeg)), dtype=np.uint8)
        dct_img = DCTImage.from_file(str(sample_jpeg))
        reconstructed = dct_img.to_pixels()

        assert reconstructed.shape == original.shape
        psnr = _psnr(original, reconstructed)
        # Our pipeline re-DCTs from Pillow's pixel decode, so double
        # quantization introduces loss. PSNR > 20dB = acceptable.
        assert psnr > 20.0, f"PSNR too low: {psnr:.1f}dB"

    def test_grayscale_roundtrip(self, grayscale_jpeg):
        original = np.array(Image.open(str(grayscale_jpeg)), dtype=np.uint8)
        dct_img = DCTImage.from_file(str(grayscale_jpeg))
        reconstructed = dct_img.to_pixels()

        assert reconstructed.shape == original.shape
        psnr = _psnr(original, reconstructed)
        assert psnr > 20.0, f"PSNR too low: {psnr:.1f}dB"

    @pytest.mark.parametrize("quality_name", ["sample_jpeg_q50", "sample_jpeg_q75", "sample_jpeg_q95"])
    def test_various_quality_factors(self, quality_name, request):
        jpeg_path = request.getfixturevalue(quality_name)
        original = np.array(Image.open(str(jpeg_path)), dtype=np.uint8)
        dct_img = DCTImage.from_file(str(jpeg_path))
        reconstructed = dct_img.to_pixels()

        psnr = _psnr(original, reconstructed)
        assert psnr > 20.0, f"PSNR too low for {quality_name}: {psnr:.1f}dB"

    def test_single_block_roundtrip(self, single_block_jpeg):
        original = np.array(Image.open(str(single_block_jpeg)), dtype=np.uint8)
        dct_img = DCTImage.from_file(str(single_block_jpeg))
        reconstructed = dct_img.to_pixels()

        h, w = original.shape[:2]
        reconstructed = reconstructed[:h, :w]
        psnr = _psnr(original, reconstructed)
        assert psnr > 10.0, f"Single block PSNR too low: {psnr:.1f}dB"


class TestSaveAndReload:
    """DCTImage -> save -> reload -> compare."""

    def test_save_reload_preserves_dimensions(self, sample_jpeg, tmp_path):
        img = read_jpeg(str(sample_jpeg))
        out = tmp_path / "saved.jpg"
        write_jpeg(img, str(out))
        img2 = read_jpeg(str(out))

        assert img2.width == img.width
        assert img2.height == img.height
        assert img2.num_components == img.num_components

    def test_save_reload_visual_quality(self, sample_jpeg, tmp_path):
        """Save and reload should produce reasonable visual quality."""
        original = np.array(Image.open(str(sample_jpeg)), dtype=np.uint8)
        img = read_jpeg(str(sample_jpeg))
        out = tmp_path / "saved.jpg"
        write_jpeg(img, str(out))

        reloaded = np.array(Image.open(str(out)), dtype=np.uint8)
        psnr = _psnr(original, reloaded)
        # Double encode introduces more loss, but should still be reasonable
        assert psnr > 20.0, f"PSNR too low after save/reload: {psnr:.1f}dB"


class TestDCModification:
    """Modify DC coefficient and verify visible change."""

    def test_brightness_increase_via_dc(self, sample_jpeg, tmp_path):
        img = DCTImage.from_file(str(sample_jpeg))
        original_pixels = img.to_pixels()

        # Increase DC coefficient (block mean) for all Y blocks
        modified_coeffs = img.y_coeffs.copy()
        modified_coeffs[:, :, 0, 0] += 10  # Increase DC

        img_bright = DCTImage(
            y_coeffs=modified_coeffs,
            cb_coeffs=img.cb_coeffs,
            cr_coeffs=img.cr_coeffs,
            quant_tables=img.quant_tables,
            width=img.width,
            height=img.height,
            comp_info=img.comp_info,
        )
        bright_pixels = img_bright.to_pixels()

        # Mean brightness should increase
        assert bright_pixels.mean() > original_pixels.mean()


class TestFormatConversion:
    """PNG/BMP -> DCTImage -> save as JPEG -> verify."""

    def test_png_to_jpeg_pipeline(self, sample_png, tmp_path):
        dct_img = convert_to_dct(str(sample_png), quality=85)
        out = tmp_path / "from_png.jpg"
        dct_img.save(str(out))

        assert out.exists()
        reloaded = Image.open(str(out))
        assert reloaded.size[0] == 256
        assert reloaded.size[1] == 256

    def test_png_conversion_quality(self, sample_png):
        """Converted image should be visually similar to original PNG."""
        original = np.array(Image.open(str(sample_png)), dtype=np.uint8)
        dct_img = convert_to_dct(str(sample_png), quality=95)
        reconstructed = dct_img.to_pixels()

        psnr = _psnr(original, reconstructed)
        assert psnr > 25.0, f"PNG conversion PSNR too low: {psnr:.1f}dB"

    def test_bmp_to_jpeg_pipeline(self, sample_bmp, tmp_path):
        dct_img = convert_to_dct(str(sample_bmp), quality=85)
        out = tmp_path / "from_bmp.jpg"
        dct_img.save(str(out))
        assert out.exists()
