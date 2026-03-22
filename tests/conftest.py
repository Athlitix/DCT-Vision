"""Shared test fixtures for dct-vision."""

import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "test_images"


@pytest.fixture
def fixture_dir():
    return FIXTURES_DIR


@pytest.fixture
def sample_jpeg():
    return FIXTURES_DIR / "natural_q85.jpg"


@pytest.fixture
def sample_jpeg_q50():
    return FIXTURES_DIR / "natural_q50.jpg"


@pytest.fixture
def sample_jpeg_q75():
    return FIXTURES_DIR / "natural_q75.jpg"


@pytest.fixture
def sample_jpeg_q95():
    return FIXTURES_DIR / "natural_q95.jpg"


@pytest.fixture
def solid_red_jpeg():
    return FIXTURES_DIR / "solid_red.jpg"


@pytest.fixture
def checkerboard_jpeg():
    return FIXTURES_DIR / "checkerboard.jpg"


@pytest.fixture
def grayscale_jpeg():
    return FIXTURES_DIR / "grayscale.jpg"


@pytest.fixture
def single_block_jpeg():
    return FIXTURES_DIR / "single_block_8x8.jpg"


@pytest.fixture
def odd_size_jpeg():
    return FIXTURES_DIR / "odd_size_100x77.jpg"


@pytest.fixture
def sub_444_jpeg():
    return FIXTURES_DIR / "sub_444.jpg"


@pytest.fixture
def sub_422_jpeg():
    return FIXTURES_DIR / "sub_422.jpg"


@pytest.fixture
def sub_420_jpeg():
    return FIXTURES_DIR / "sub_420.jpg"


@pytest.fixture
def sample_png():
    return FIXTURES_DIR / "sample.png"


@pytest.fixture
def sample_bmp():
    return FIXTURES_DIR / "sample.bmp"
