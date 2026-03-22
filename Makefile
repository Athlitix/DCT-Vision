.PHONY: test lint bench bench-quick coverage all setup

UV = UV_PROJECT_ENVIRONMENT=dv-env uv run

# First-time setup
setup:
	uv venv dv-env --python 3.12
	UV_PROJECT_ENVIRONMENT=dv-env uv sync --all-groups

# Run all checks
all: test

# Tests
test:
	$(UV) pytest tests/ -v --tb=short

coverage:
	$(UV) pytest tests/ --cov=dct_vision --cov-report=html --cov-report=term --cov-fail-under=90

# Benchmarks
bench:
	$(UV) python benchmarks/run_all.py --output benchmarks/results/latest.json --repeats 20

bench-quick:
	$(UV) python benchmarks/run_all.py --repeats 5 --resolutions 256,512

# CLI smoke test
cli-test:
	$(UV) dv --version
	$(UV) dv info tests/fixtures/test_images/natural_q85.jpg
	$(UV) dv blur tests/fixtures/test_images/natural_q85.jpg -o /tmp/dv_test_blur.jpg --sigma 2.0
