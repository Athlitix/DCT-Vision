"""DCT-Vision command line interface."""

import json
import time
import tracemalloc
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from dct_vision import __version__

app = typer.Typer(
    name="dv",
    help="DCT-Vision: Frequency-domain native image processing.",
    add_completion=True,
    no_args_is_help=True,
)


def version_callback(value: bool):
    if value:
        typer.echo(f"dct-vision {__version__}")
        raise typer.Exit()


# Global state for verbose/quiet
_verbose_level = 0
_quiet = False


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=version_callback, is_eager=True,
        help="Show version and exit.",
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True,
        help="Increase log level (stackable: -vv for DEBUG).",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Suppress all output except errors.",
    ),
):
    """DCT-Vision: Frequency-domain native image processing."""
    global _verbose_level, _quiet
    _verbose_level = verbose
    _quiet = quiet

    if verbose > 0:
        import logging
        from dct_vision.utils.logging import configure_logging
        level = logging.DEBUG if verbose >= 2 else logging.INFO
        configure_logging(level)


class EdgeMethod(str, Enum):
    laplacian = "laplacian"
    gradient = "gradient"


class ChannelChoice(str, Enum):
    all = "all"
    luma = "luma"
    chroma = "chroma"


class FlipDirection(str, Enum):
    horizontal = "horizontal"
    vertical = "vertical"


def _echo(msg: str):
    """Print unless quiet mode."""
    if not _quiet:
        typer.echo(msg)


def _load_image(path: Path):
    """Load a JPEG or convert from other format."""
    from dct_vision.io.validation import detect_format
    from dct_vision.core.dct_image import DCTImage
    from dct_vision.io.convert import convert_to_dct

    fmt = detect_format(str(path))
    if fmt == "jpeg":
        return DCTImage.from_file(str(path))
    else:
        return convert_to_dct(str(path))


def _timed(func):
    """Run a function and return (result, elapsed_ms)."""
    start = time.perf_counter()
    result = func()
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed


def _measure_memory(func):
    """Run function and return (result, peak_kb)."""
    tracemalloc.start()
    result = func()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1024


# -- Image Operations --

@app.command()
def blur(
    input_path: Path = typer.Argument(..., help="Input image path.", exists=True),
    output: Path = typer.Option(..., "--output", "-o", help="Output image path."),
    sigma: float = typer.Option(2.0, "--sigma", "-s", help="Gaussian blur sigma."),
    channels: ChannelChoice = typer.Option(ChannelChoice.all, "--channels", help="Channels to process."),
    cross_block: bool = typer.Option(False, "--cross-block", help="Use cross-block strategy (better quality for sigma > 2)."),
    timing: bool = typer.Option(False, "--timing", "-t", help="Print execution time."),
    memory: bool = typer.Option(False, "--memory", help="Print peak memory usage."),
):
    """Apply Gaussian blur in DCT domain."""
    from dct_vision.ops.blur import blur as blur_op

    if sigma <= 0:
        typer.echo("Error: sigma must be > 0", err=True)
        raise typer.Exit(code=1)

    img = _load_image(input_path)

    def run():
        return blur_op(img, sigma=sigma, channels=channels.value, cross_block=cross_block)

    if memory:
        result, peak_kb = _measure_memory(run)
    else:
        result, elapsed = _timed(run)

    result.save(str(output))
    _echo(f"Blurred {input_path.name} -> {output.name} (sigma={sigma}, channels={channels.value})")
    if timing and not memory:
        _echo(f"Time: {elapsed:.1f}ms")
    if memory:
        _echo(f"Peak memory: {peak_kb:.1f}KB")


@app.command()
def sharpen(
    input_path: Path = typer.Argument(..., help="Input image path.", exists=True),
    output: Path = typer.Option(..., "--output", "-o", help="Output image path."),
    amount: float = typer.Option(1.5, "--amount", "-a", help="Sharpening strength."),
    timing: bool = typer.Option(False, "--timing", "-t", help="Print execution time."),
    memory: bool = typer.Option(False, "--memory", help="Print peak memory usage."),
):
    """Sharpen image by boosting high-frequency DCT coefficients."""
    from dct_vision.ops.sharpen import sharpen as sharpen_op

    if amount <= 0:
        typer.echo("Error: amount must be > 0", err=True)
        raise typer.Exit(code=1)

    img = _load_image(input_path)
    result, elapsed = _timed(lambda: sharpen_op(img, amount=amount))
    result.save(str(output))

    _echo(f"Sharpened {input_path.name} -> {output.name} (amount={amount})")
    if timing:
        _echo(f"Time: {elapsed:.1f}ms")


@app.command()
def brightness(
    input_path: Path = typer.Argument(..., help="Input image path.", exists=True),
    output: Path = typer.Option(..., "--output", "-o", help="Output image path."),
    offset: float = typer.Option(30.0, "--offset", help="Brightness offset."),
    timing: bool = typer.Option(False, "--timing", "-t", help="Print execution time."),
):
    """Adjust brightness by modifying DC coefficients."""
    from dct_vision.ops.color import adjust_brightness

    img = _load_image(input_path)
    result, elapsed = _timed(lambda: adjust_brightness(img, offset=offset))
    result.save(str(output))

    _echo(f"Brightness adjusted {input_path.name} -> {output.name} (offset={offset})")
    if timing:
        _echo(f"Time: {elapsed:.1f}ms")


@app.command()
def contrast(
    input_path: Path = typer.Argument(..., help="Input image path.", exists=True),
    output: Path = typer.Option(..., "--output", "-o", help="Output image path."),
    factor: float = typer.Option(1.5, "--factor", "-f", help="Contrast factor."),
    timing: bool = typer.Option(False, "--timing", "-t", help="Print execution time."),
):
    """Adjust contrast by scaling AC coefficients."""
    from dct_vision.ops.color import adjust_contrast

    if factor < 0:
        typer.echo("Error: factor must be >= 0", err=True)
        raise typer.Exit(code=1)

    img = _load_image(input_path)
    result, elapsed = _timed(lambda: adjust_contrast(img, factor=factor))
    result.save(str(output))

    _echo(f"Contrast adjusted {input_path.name} -> {output.name} (factor={factor})")
    if timing:
        _echo(f"Time: {elapsed:.1f}ms")


@app.command()
def downscale(
    input_path: Path = typer.Argument(..., help="Input image path.", exists=True),
    output: Path = typer.Option(..., "--output", "-o", help="Output image path."),
    factor: int = typer.Option(2, "--factor", help="Downscale factor (power of 2)."),
    timing: bool = typer.Option(False, "--timing", "-t", help="Print execution time."),
):
    """Downscale image via DCT frequency truncation."""
    from dct_vision.ops.scale import downscale as downscale_op

    if factor < 1 or (factor & (factor - 1)) != 0:
        typer.echo("Error: factor must be a positive power of 2", err=True)
        raise typer.Exit(code=1)

    img = _load_image(input_path)
    result, elapsed = _timed(lambda: downscale_op(img, factor=factor))
    result.save(str(output))

    _echo(
        f"Downscaled {input_path.name} -> {output.name} "
        f"({img.width}x{img.height} -> {result.width}x{result.height})"
    )
    if timing:
        _echo(f"Time: {elapsed:.1f}ms")


@app.command()
def edges(
    input_path: Path = typer.Argument(..., help="Input image path.", exists=True),
    output: Path = typer.Option(..., "--output", "-o", help="Output image path."),
    method: EdgeMethod = typer.Option(EdgeMethod.laplacian, "--method", "-m", help="Edge detection method."),
    timing: bool = typer.Option(False, "--timing", "-t", help="Print execution time."),
):
    """Detect edges in DCT domain."""
    from dct_vision.ops.edge import detect_edges

    img = _load_image(input_path)
    result, elapsed = _timed(lambda: detect_edges(img, method=method.value))
    result.save(str(output))

    _echo(f"Edge detection ({method.value}) {input_path.name} -> {output.name}")
    if timing:
        _echo(f"Time: {elapsed:.1f}ms")


# -- Format Conversion --

@app.command()
def convert(
    input_path: Path = typer.Argument(..., help="Input image path.", exists=True),
    output: Path = typer.Option(..., "--output", "-o", help="Output JPEG path."),
    quality: int = typer.Option(85, "--quality", "-q", help="JPEG quality (1-100)."),
    timing: bool = typer.Option(False, "--timing", "-t", help="Print execution time."),
):
    """Convert PNG/BMP/TIFF to JPEG via DCT representation."""
    from dct_vision.io.convert import convert_to_dct

    img, elapsed = _timed(lambda: convert_to_dct(str(input_path), quality=quality))
    img.save(str(output))

    _echo(f"Converted {input_path.name} -> {output.name} (quality={quality})")
    if timing:
        _echo(f"Time: {elapsed:.1f}ms")


# -- Inspection / Analysis --

@app.command()
def info(
    input_path: Path = typer.Argument(..., help="Input image path.", exists=True),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
):
    """Display DCT image information."""
    from dct_vision.ops.quality import estimate_quality, dct_stats

    img = _load_image(input_path)
    q = estimate_quality(img)
    stats = dct_stats(img)

    # Determine subsampling
    subsampling = "unknown"
    if img.comp_info and len(img.comp_info) >= 1:
        h = img.comp_info[0].get("h_samp_factor", 1)
        v = img.comp_info[0].get("v_samp_factor", 1)
        if h == 1 and v == 1:
            subsampling = "4:4:4"
        elif h == 2 and v == 1:
            subsampling = "4:2:2"
        elif h == 2 and v == 2:
            subsampling = "4:2:0"

    data = {
        "file": str(input_path),
        "width": img.width,
        "height": img.height,
        "components": img.num_components,
        "subsampling": subsampling,
        "blocks_y": f"{img.y_coeffs.shape[0]}x{img.y_coeffs.shape[1]}",
        "estimated_quality": q,
        "dc_mean": round(stats["dc_mean"], 2),
        "dc_std": round(stats["dc_std"], 2),
        "ac_energy": round(stats["ac_energy"], 2),
        "nonzero_ac_ratio": round(stats["num_nonzero_ac"] / max(stats["total_ac"], 1), 4),
    }

    if json_output:
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(f"File:            {data['file']}")
        typer.echo(f"Dimensions:      {data['width']}x{data['height']}")
        typer.echo(f"Components:      {data['components']}")
        typer.echo(f"Subsampling:     {data['subsampling']}")
        typer.echo(f"Y blocks:        {data['blocks_y']}")
        typer.echo(f"Est. quality:    {data['estimated_quality']}")
        typer.echo(f"DC mean/std:     {data['dc_mean']} / {data['dc_std']}")
        typer.echo(f"AC energy:       {data['ac_energy']}")
        typer.echo(f"Nonzero AC:      {data['nonzero_ac_ratio']:.2%}")


@app.command()
def inspect(
    input_path: Path = typer.Argument(..., help="Input image path.", exists=True),
    block: str = typer.Option(..., "--block", "-b", help="Block coordinates as 'row,col'."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
):
    """Dump DCT coefficients for a specific block."""
    import numpy as np

    try:
        row, col = [int(x.strip()) for x in block.split(",")]
    except ValueError:
        typer.echo("Error: --block must be 'row,col' (e.g. --block 5,10)", err=True)
        raise typer.Exit(code=1)

    img = _load_image(input_path)

    if row >= img.y_coeffs.shape[0] or col >= img.y_coeffs.shape[1]:
        typer.echo(
            f"Error: block ({row},{col}) out of range. "
            f"Y blocks: {img.y_coeffs.shape[0]}x{img.y_coeffs.shape[1]}",
            err=True,
        )
        raise typer.Exit(code=1)

    y_block = img.y_coeffs[row, col].tolist()

    data = {
        "block": [row, col],
        "y_coefficients": y_block,
    }
    if img.cb_coeffs is not None:
        cb_bh, cb_bw = img.cb_coeffs.shape[:2]
        cb_r = min(row * cb_bh // img.y_coeffs.shape[0], cb_bh - 1)
        cb_c = min(col * cb_bw // img.y_coeffs.shape[1], cb_bw - 1)
        data["cb_coefficients"] = img.cb_coeffs[cb_r, cb_c].tolist()
        data["cr_coefficients"] = img.cr_coeffs[cb_r, cb_c].tolist()

    if json_output:
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(f"Block ({row}, {col}) - Y channel DCT coefficients:")
        y = np.array(y_block)
        for r in range(8):
            vals = "  ".join(f"{int(v):6d}" for v in y[r])
            typer.echo(f"  {vals}")


@app.command()
def quality(
    input_path: Path = typer.Argument(..., help="Input image path.", exists=True),
):
    """Estimate JPEG quality factor from quantization tables."""
    from dct_vision.ops.quality import estimate_quality

    img = _load_image(input_path)
    q = estimate_quality(img)
    typer.echo(f"Estimated JPEG quality: {q}")


# -- Augmentation --

@app.command()
def augment(
    input_path: Path = typer.Argument(..., help="Input image path.", exists=True),
    output: Path = typer.Option(..., "--output", "-o", help="Output image path."),
    flip: Optional[FlipDirection] = typer.Option(None, "--flip", help="Flip direction."),
    brightness_jitter_val: float = typer.Option(0.0, "--brightness-jitter", help="Max brightness offset."),
    contrast_jitter_val: float = typer.Option(0.0, "--contrast-jitter", help="Max contrast deviation."),
    noise_sigma: float = typer.Option(0.0, "--noise", help="Gaussian noise sigma."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed."),
    timing: bool = typer.Option(False, "--timing", "-t", help="Print execution time."),
):
    """Apply augmentations in DCT domain (for ML pipelines)."""
    from dct_vision.augment.flip import horizontal_flip, vertical_flip
    from dct_vision.augment.jitter import brightness_jitter, contrast_jitter
    from dct_vision.augment.noise import gaussian_noise

    img = _load_image(input_path)
    start = time.perf_counter()

    if flip == FlipDirection.horizontal:
        img = horizontal_flip(img)
    elif flip == FlipDirection.vertical:
        img = vertical_flip(img)

    if brightness_jitter_val > 0:
        img = brightness_jitter(img, max_offset=brightness_jitter_val, seed=seed)

    if contrast_jitter_val > 0:
        img = contrast_jitter(img, max_factor=contrast_jitter_val, seed=seed)

    if noise_sigma > 0:
        img = gaussian_noise(img, sigma=noise_sigma, seed=seed)

    elapsed = (time.perf_counter() - start) * 1000
    img.save(str(output))

    applied = []
    if flip:
        applied.append(f"flip={flip.value}")
    if brightness_jitter_val > 0:
        applied.append(f"brightness_jitter={brightness_jitter_val}")
    if contrast_jitter_val > 0:
        applied.append(f"contrast_jitter={contrast_jitter_val}")
    if noise_sigma > 0:
        applied.append(f"noise={noise_sigma}")

    _echo(f"Augmented {input_path.name} -> {output.name} ({', '.join(applied) or 'no-op'})")
    if timing:
        _echo(f"Time: {elapsed:.1f}ms")
