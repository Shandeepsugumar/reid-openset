.PHONY: help install lint test train eval export docker-build

PYTHON ?= python
PIP ?= pip

help:
	@echo "Available make targets:"
	@echo "  install      - Install dependencies and local package in editable mode"
	@echo "  lint         - Run ruff check and black code formatting verification"
	@echo "  test         - Run unit and integration tests with pytest"
	@echo "  train        - Run open-set person ReID model training"
	@echo "  eval         - Run open-set evaluation benchmarks"
	@echo "  export       - Export trained model to ONNX runtime format"
	@echo "  docker-build - Build Docker deployment image"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

lint:
	$(PYTHON) -m ruff check src tests configs scripts service
	$(PYTHON) -m black --check src tests configs scripts service

test:
	$(PYTHON) -m pytest tests/

train:
	$(PYTHON) scripts/train.py --config configs/default.yaml

eval:
	$(PYTHON) scripts/eval.py --config configs/default.yaml

export:
	$(PYTHON) scripts/export.py --config configs/default.yaml

docker-build:
	docker build -t reid-openset:latest -f docker/Dockerfile .
