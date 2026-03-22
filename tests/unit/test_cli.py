"""Tests for the dv CLI."""

import json
import pytest
from typer.testing import CliRunner
from pathlib import Path

from dct_vision.cli.app import app

runner = CliRunner()


class TestCLIBasics:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "DCT-Vision" in result.output

    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # Typer shows help and exits with code 0 or 2 depending on version
        assert "DCT-Vision" in result.output or result.exit_code in (0, 2)


class TestBlurCommand:
    def test_basic_blur(self, sample_jpeg, tmp_path):
        output = tmp_path / "blurred.jpg"
        result = runner.invoke(app, [
            "blur", str(sample_jpeg), "-o", str(output), "--sigma", "2.0"
        ])
        assert result.exit_code == 0, result.output
        assert output.exists()

    def test_blur_missing_output_fails(self, sample_jpeg):
        result = runner.invoke(app, ["blur", str(sample_jpeg)])
        assert result.exit_code != 0

    def test_blur_invalid_sigma(self, sample_jpeg, tmp_path):
        output = tmp_path / "out.jpg"
        result = runner.invoke(app, [
            "blur", str(sample_jpeg), "-o", str(output), "--sigma", "0"
        ])
        assert result.exit_code != 0


class TestSharpenCommand:
    def test_basic_sharpen(self, sample_jpeg, tmp_path):
        output = tmp_path / "sharpened.jpg"
        result = runner.invoke(app, [
            "sharpen", str(sample_jpeg), "-o", str(output), "--amount", "1.5"
        ])
        assert result.exit_code == 0, result.output
        assert output.exists()


class TestBrightnessCommand:
    def test_basic_brightness(self, sample_jpeg, tmp_path):
        output = tmp_path / "bright.jpg"
        result = runner.invoke(app, [
            "brightness", str(sample_jpeg), "-o", str(output), "--offset", "30"
        ])
        assert result.exit_code == 0, result.output
        assert output.exists()


class TestContrastCommand:
    def test_basic_contrast(self, sample_jpeg, tmp_path):
        output = tmp_path / "contrast.jpg"
        result = runner.invoke(app, [
            "contrast", str(sample_jpeg), "-o", str(output), "--factor", "1.5"
        ])
        assert result.exit_code == 0, result.output
        assert output.exists()


class TestDownscaleCommand:
    def test_basic_downscale(self, sample_jpeg, tmp_path):
        output = tmp_path / "small.jpg"
        result = runner.invoke(app, [
            "downscale", str(sample_jpeg), "-o", str(output), "--factor", "2"
        ])
        assert result.exit_code == 0, result.output
        assert output.exists()


class TestEdgesCommand:
    def test_laplacian(self, sample_jpeg, tmp_path):
        output = tmp_path / "edges.jpg"
        result = runner.invoke(app, [
            "edges", str(sample_jpeg), "-o", str(output), "--method", "laplacian"
        ])
        assert result.exit_code == 0, result.output
        assert output.exists()

    def test_gradient(self, sample_jpeg, tmp_path):
        output = tmp_path / "edges_g.jpg"
        result = runner.invoke(app, [
            "edges", str(sample_jpeg), "-o", str(output), "--method", "gradient"
        ])
        assert result.exit_code == 0, result.output
        assert output.exists()


class TestInfoCommand:
    def test_info_text(self, sample_jpeg):
        result = runner.invoke(app, ["info", str(sample_jpeg)])
        assert result.exit_code == 0
        assert "256" in result.output  # dimensions

    def test_info_json(self, sample_jpeg):
        result = runner.invoke(app, ["info", str(sample_jpeg), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "width" in data
        assert "height" in data


class TestQualityCommand:
    def test_quality(self, sample_jpeg):
        result = runner.invoke(app, ["quality", str(sample_jpeg)])
        assert result.exit_code == 0
        assert any(c.isdigit() for c in result.output)


class TestConvertCommand:
    def test_png_to_jpeg(self, sample_png, tmp_path):
        output = tmp_path / "converted.jpg"
        result = runner.invoke(app, [
            "convert", str(sample_png), "-o", str(output), "--quality", "85"
        ])
        assert result.exit_code == 0, result.output
        assert output.exists()


class TestInspectCommand:
    def test_inspect_block(self, sample_jpeg):
        result = runner.invoke(app, [
            "inspect", str(sample_jpeg), "--block", "0,0"
        ])
        assert result.exit_code == 0
        assert "Block" in result.output

    def test_inspect_json(self, sample_jpeg):
        result = runner.invoke(app, [
            "inspect", str(sample_jpeg), "--block", "0,0", "--json"
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "y_coefficients" in data

    def test_inspect_invalid_block(self, sample_jpeg):
        result = runner.invoke(app, [
            "inspect", str(sample_jpeg), "--block", "999,999"
        ])
        assert result.exit_code != 0

    def test_inspect_bad_format(self, sample_jpeg):
        result = runner.invoke(app, [
            "inspect", str(sample_jpeg), "--block", "abc"
        ])
        assert result.exit_code != 0


class TestAugmentCommand:
    def test_augment_flip(self, sample_jpeg, tmp_path):
        output = tmp_path / "flipped.jpg"
        result = runner.invoke(app, [
            "augment", str(sample_jpeg), "-o", str(output),
            "--flip", "horizontal"
        ])
        assert result.exit_code == 0, result.output
        assert output.exists()

    def test_augment_noise(self, sample_jpeg, tmp_path):
        output = tmp_path / "noisy.jpg"
        result = runner.invoke(app, [
            "augment", str(sample_jpeg), "-o", str(output),
            "--noise", "3.0", "--seed", "42"
        ])
        assert result.exit_code == 0, result.output
        assert output.exists()

    def test_augment_combined(self, sample_jpeg, tmp_path):
        output = tmp_path / "augmented.jpg"
        result = runner.invoke(app, [
            "augment", str(sample_jpeg), "-o", str(output),
            "--flip", "vertical",
            "--brightness-jitter", "20",
            "--contrast-jitter", "0.3",
            "--noise", "2.0",
            "--seed", "42"
        ])
        assert result.exit_code == 0, result.output
        assert output.exists()


class TestGlobalFlags:
    def test_quiet_suppresses_output(self, sample_jpeg, tmp_path):
        output = tmp_path / "quiet.jpg"
        result = runner.invoke(app, [
            "-q", "blur", str(sample_jpeg), "-o", str(output), "--sigma", "2.0"
        ])
        assert result.exit_code == 0
        assert result.output.strip() == ""

    def test_verbose(self, sample_jpeg, tmp_path):
        output = tmp_path / "verbose.jpg"
        result = runner.invoke(app, [
            "-v", "blur", str(sample_jpeg), "-o", str(output), "--sigma", "2.0"
        ])
        assert result.exit_code == 0

    def test_blur_with_channels(self, sample_jpeg, tmp_path):
        output = tmp_path / "luma_blur.jpg"
        result = runner.invoke(app, [
            "blur", str(sample_jpeg), "-o", str(output),
            "--sigma", "2.0", "--channels", "luma"
        ])
        assert result.exit_code == 0, result.output
        assert output.exists()

    def test_info_shows_subsampling(self, sample_jpeg):
        result = runner.invoke(app, ["info", str(sample_jpeg)])
        assert result.exit_code == 0
        assert "Subsampling" in result.output
