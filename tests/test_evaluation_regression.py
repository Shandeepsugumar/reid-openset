"""Regression test for baseline ResNet-18 evaluation metrics against reference artifacts."""

from pathlib import Path
import pytest
from scripts.evaluate_embeddings import evaluate_embeddings

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "baseline_resnet18"
REQUIRED_FILES = [
    "query_embeddings.npy",
    "query_pids.npy",
    "query_camids.npy",
    "gallery_embeddings.npy",
    "gallery_pids.npy",
    "gallery_camids.npy",
]


@pytest.mark.skipif(
    not ARTIFACTS_DIR.is_dir()
    or not all((ARTIFACTS_DIR / f).is_file() for f in REQUIRED_FILES),
    reason="Baseline evaluation artifacts (.npy files) not found on disk",
)
def test_baseline_resnet18_regression() -> None:
    """Verify that evaluator reproduces reference Colab baseline metrics within 0.05 points."""
    metrics = evaluate_embeddings(
        embeddings_dir=ARTIFACTS_DIR,
        known_ratio=0.5,
        seed=42,
    )

    # Reference scores from Colab T4 baseline:
    # Rank-1: 90.23%, mAP: 76.89%, FAR@TAR=80%: 14.42%, FAR@TAR=90%: 37.37%
    assert abs(metrics["rank_1"] - 90.23) <= 0.05, f"Rank-1 {metrics['rank_1']} deviated from 90.23"
    assert abs(metrics["mAP"] - 76.89) <= 0.05, f"mAP {metrics['mAP']} deviated from 76.89"
    assert abs(metrics["far_at_tar_80"] - 14.42) <= 0.05, f"FAR@80 {metrics['far_at_tar_80']} deviated from 14.42"
    assert abs(metrics["far_at_tar_90"] - 37.37) <= 0.05, f"FAR@90 {metrics['far_at_tar_90']} deviated from 37.37"
