# Installation

## Requirements

- Python 3.10+
- libjpeg-turbo (runtime library, usually pre-installed on Linux)

## Install from PyPI

```bash
pip install dct-vision
```

## Install from source

```bash
git clone https://github.com/athlitix/dct-vision.git
cd dct-vision
pip install -e .
```

## Install with uv (recommended for development)

```bash
git clone https://github.com/athlitix/dct-vision.git
cd dct-vision
uv venv dv-env --python 3.12
UV_PROJECT_ENVIRONMENT=dv-env uv sync
```

## System dependencies

The native libjpeg bindings require `libjpeg-turbo` runtime library:

```bash
# Ubuntu/Debian
sudo apt install libjpeg-turbo8

# For development (optional, enables native DCT extraction)
sudo apt install libjpeg-turbo8-dev
```

If libjpeg-turbo is not available, the library falls back to Pillow-based extraction automatically.

## Verify installation

```bash
dv --version
dv info path/to/image.jpg
```
