.PHONY: test lint bench bench-quick bench-crossblock ml-bench coverage all setup

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
	$(UV) python benchmarks/run_all.py --output benchmarks/results/latest.json \
		--repeats 100 --resolutions 256,512,1024,2048 --qualities 50,75,85,95

bench-quick:
	$(UV) python benchmarks/run_all.py --repeats 5 --resolutions 256,512 --no-e2e

bench-crossblock:
	$(UV) python benchmarks/cross_block_sweep.py --output benchmarks/results/cross_block.json

# ML validation (CIFAR-10): pixel vs DCT y_only vs DCT ycbcr
ml-bench:
	$(UV) python -m dct_vision.ml.train --models pixelcnn,dctcnn,dctfreq \
		--epochs 15 -o benchmarks/results/ml2_cifar.json
	$(UV) python -m dct_vision.ml.train --models dctcnn --mode ycbcr \
		--epochs 15 -o benchmarks/results/ml2_ycbcr.json
	$(UV) python -m dct_vision.ml.train --models pixelresnet,dctresnet --augment \
		--epochs 20 -o benchmarks/results/ml3_resnet_aug.json

# CLI smoke test
cli-test:
	$(UV) dv --version
	$(UV) dv info tests/fixtures/test_images/natural_q85.jpg
	$(UV) dv blur tests/fixtures/test_images/natural_q85.jpg -o /tmp/dv_test_blur.jpg --sigma 2.0
