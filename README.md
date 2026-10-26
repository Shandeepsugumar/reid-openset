# reid-openset: Open-Set Person Re-Identification

A modular Python framework for open-set person re-identification (ReID). Unlike standard closed-set ReID which assumes all query identities exist in the gallery, this framework incorporates rejection mechanisms, confidence calibration, and open-set evaluation metrics to identify known gallery subjects while reliably detecting unseen "impostor" identities.

---

## Project Structure

```
reid-openset/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI workflow
├── configs/                     # YAML configuration files (models, data, training, eval)
├── docker/
│   └── Dockerfile               # Production & inference container specification
├── docs/                        # Architecture notes and documentation
├── scripts/                     # Executable scripts for training, evaluation, and ONNX export
├── service/                     # FastAPI inference microservice
├── src/
│   └── reid/
│       ├── calibration/         # Uncertainty quantification, thresholding, and temperature scaling
│       ├── data/                # Datasets, transforms, samplers, and dataloaders
│       ├── eval/                # Open-set metrics (DIR @ FAR, CMC, mAP, ROC, Detection & Identification)
│       ├── losses/              # Loss functions (Cross-entropy, Triplet, ArcFace, Open-set margin losses)
│       └── models/              # Backbone architectures, pooling heads, and ReID feature extractors
├── tests/                       # Unit and integration test suite
├── .gitignore                   # Ignored files and directories
├── Makefile                     # Build, test, and workflow automation
├── pyproject.toml               # Package metadata and tool configurations
├── README.md                    # Project overview and documentation
└── requirements.txt             # Project dependencies
```

---

## Core Components

- **`src/reid/models`**: Feature extraction backbones and embedding heads for identity representation.
- **`src/reid/losses`**: Metric learning and classification objectives suitable for open-set separability.
- **`src/reid/data`**: Preprocessing pipelines, data augmentations, and identity-aware batch sampling.
- **`src/reid/calibration`**: Open-set score calibration (e.g., Extreme Value Theory / EVT, Platt scaling, cosine thresholding).
- **`src/reid/eval`**: Evaluation protocols for open-set person ReID (Detection and Identification Rate vs. False Alarm Rate).
- **`service/`**: Real-time feature extraction and gallery matching service via FastAPI.

---

## Dataset

> [!IMPORTANT]
> The **Market-1501** dataset is **NOT** included in this repository and must not be redistributed.

### Obtaining the Data
Obtain the dataset from the official release (or the Kaggle copy).

### Directory Placement
Extract or link the dataset into the repository under:
```text
data/Market-1501-v15.09.15/
├── bounding_box_train/   # Training identity bounding boxes
├── bounding_box_test/    # Gallery images (including distractors & junk)
├── query/                # Probe / query images
├── gt_bbox/              # (optional) Ground-truth bounding boxes
├── gt_query/             # (optional) Ground-truth query labels
└── readme.txt
```

### Custom Data Path Configuration
If your dataset is located in another folder (or an external drive), point to it using the `REID_DATA_ROOT` environment variable or the `--data-dir` argument:

```bash
# Via environment variable
export REID_DATA_ROOT="/path/to/Market-1501-v15.09.15"

# Or on Windows PowerShell:
$env:REID_DATA_ROOT = "D:\path\to\Market-1501-v15.09.15"

# Or via command line override
python scripts/train.py --data-dir "/path/to/Market-1501-v15.09.15"
```

---

## Quick Start

### 1. Installation

```bash
# Clone the repository and navigate into the project root
git clone <repo-url>
cd reid-openset

# Install dependencies and local package in editable mode
make install
```

### 2. Code Quality & Testing

```bash
# Run linting and style checks
make lint

# Run unit tests
make test
```

### 3. Training & Evaluation Workflow

```bash
# Train an open-set ReID model
make train

# Evaluate on open-set benchmark protocols
make eval

# Export the trained model to ONNX
make export
```

### 4. Docker Deployment

```bash
# Build the Docker image
make docker-build
```

---

## License

This project is licensed under the MIT License.
