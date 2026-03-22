"""DCT-Vision command line interface."""

import json
import sys
import time
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


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=version_callback, is_eager=True,
        help="Show version and exit.",
    ),
):
    """DCT-Vision: Frequency-domain native image processing."""


class EdgeMethod(str, Enum):
    laplacian = "laplacian"
    gradient = "gradient"


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


# -- Image Operations --

@app.command()
def blur(
    input_path: Path = typer.Argument(..., help="Input image path.", exists=True),
    output: Path = typer.Option(..., "--output", "-o", help="Output image path."),
    sigma: float = typer.Option(2.0, "--sigma", "-s", help="Gaussian blur sigma."),
    timing: bool = typer.Option(False, "--timing", "-t", help="Print execution time."),
):
    """Apply Gaussian blur in DCT domain."""
    from dct_vision.ops.blur import blur as blur_op

    if sigma <= 0:
        typer.echo("Error: sigma must be > 0", err=True)
        raise typer.Exit(code=1)

    img = _load_image(input_path)
    result, elapsed = _timed(lambda: blur_op(img, sigma=sigma))
    result.save(str(output))

    typer.echo(f"Blurred {input_path.name} -> {output.name} (sigma={sigma})")
    if timing:
        typer.echo(f"Time: {elapsed:.1f}ms")


@app.command()
def sharpen(
    input_path: Path = typer.Argument(..., help="Input image path.", exists=True),
    output: Path = typer.Option(..., "--output", "-o", help="Output image path."),
    amount: float = typer.Option(1.5, "--amount", "-a", help="Sharpening strength."),
    timing: bool = typer.Option(False, "--timing", "-t", help="Print execution time."),
):
    """Sharpen image by boosting high-frequency DCT coefficients."""
    from dct_vision.ops.sharpen import sharpen as sharpen_op

    if amount <= 0:
        typer.echo("Error: amount must be > 0", err=True)
        raise typer.Exit(code=1)

    img = _load_image(input_path)
    result, elapsed = _timed(lambda: sharpen_op(img, amount=amount))
    result.save(str(output))

    typer.echo(f"Sharpened {input_path.name} -> {output.name} (amount={amount})")
    if timing:
        typer.echo(f"Time: {elapsed:.1f}ms")


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

    typer.echo(f"Brightness adjusted {input_path.name} -> {output.name} (offset={offset})")
    if timing:
        typer.echo(f"Time: {elapsed:.1f}ms")


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

    typer.echo(f"Contrast adjusted {input_path.name} -> {output.name} (factor={factor})")
    if timing:
        typer.echo(f"Time: {elapsed:.1f}ms")


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

    typer.echo(
        f"Downscaled {input_path.name} -> {output.name} "
        f"({img.width}x{img.height} -> {result.width}x{result.height})"
    )
    if timing:
        typer.echo(f"Time: {elapsed:.1f}ms")


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

    typer.echo(f"Edge detection ({method.value}) {input_path.name} -> {output.name}")
    if timing:
        typer.echo(f"Time: {elapsed:.1f}ms")


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

    data = {
        "file": str(input_path),
        "width": img.width,
        "height": img.height,
        "components": img.num_components,
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
        typer.echo(f"Y blocks:        {data['blocks_y']}")
        typer.echo(f"Est. quality:    {data['estimated_quality']}")
        typer.echo(f"DC mean/std:     {data['dc_mean']} / {data['dc_std']}")
        typer.echo(f"AC energy:       {data['ac_energy']}")
        typer.echo(f"Nonzero AC:      {data['nonzero_ac_ratio']:.2%}")


@app.command()
def quality(
    input_path: Path = typer.Argument(..., help="Input image path.", exists=True),
):
    """Estimate JPEG quality factor from quantization tables."""
    from dct_vision.ops.quality import estimate_quality

    img = _load_image(input_path)
    q = estimate_quality(img)
    typer.echo(f"Estimated JPEG quality: {q}")
