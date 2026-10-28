"""Unit tests for training utilities, checkpoint resumption, and CSV logging."""

from pathlib import Path

import pytest
import torch

from reid.models import ReIDBackbone
from scripts.train import CSVMetricLogger, compute_config_hash, load_config, train


def test_load_config() -> None:
    cfg = load_config("configs/default.yaml")
    assert "model" in cfg
    assert "data" in cfg
    assert "loss" in cfg
    assert "training" in cfg
    assert cfg["model"]["backbone"] == "resnet18"
    assert cfg["model"]["embedding_dim"] == 512


def test_csv_metric_logger(tmp_path: Path) -> None:
    csv_file = tmp_path / "metrics.csv"
    logger = CSVMetricLogger(csv_file, resume=False)

    metrics_epoch_1 = {
        "epoch": 1,
        "lr": 0.00035,
        "loss": 4.12,
        "loss_ce": 3.5,
        "loss_triplet": 0.62,
        "dist_ap": 0.25,
        "dist_an": 0.85,
        "epoch_time_sec": 12.3,
    }
    logger.log_epoch(metrics_epoch_1)

    assert csv_file.is_file()
    lines = csv_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # header + 1 epoch

    # Resume test
    logger_resumed = CSVMetricLogger(csv_file, resume=True)
    metrics_epoch_2 = {
        "epoch": 2,
        "lr": 0.00035,
        "loss": 3.80,
        "loss_ce": 3.2,
        "loss_triplet": 0.60,
        "dist_ap": 0.22,
        "dist_an": 0.88,
        "epoch_time_sec": 11.9,
    }
    logger_resumed.log_epoch(metrics_epoch_2)
    lines_after = csv_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines_after) == 3  # header + 2 epochs


def test_model_checkpoint_resumption(tmp_path: Path) -> None:
    model1 = ReIDBackbone(backbone_name="resnet18", embedding_dim=512, num_classes=50)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3)

    ckpt_path = tmp_path / "latest_checkpoint.pth"
    torch.save(
        {
            "epoch": 5,
            "model_state_dict": model1.state_dict(),
            "optimizer_state_dict": optimizer1.state_dict(),
        },
        ckpt_path,
    )

    # Resume into fresh model
    model2 = ReIDBackbone(backbone_name="resnet18", embedding_dim=512, num_classes=50)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model2.load_state_dict(checkpoint["model_state_dict"])
    resumed_epoch = checkpoint["epoch"] + 1

    assert resumed_epoch == 6

    # Verify weights match
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_config_hash_mismatch_refuses_to_resume(tmp_path: Path) -> None:
    """Verify that train() raises ValueError if checkpoint was saved with different config hash."""
    exp_name = "test_hash_exp"
    output_root = tmp_path / "outputs"
    ckpt_dir = output_root / exp_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt = ckpt_dir / "latest_checkpoint.pth"
    torch.save(
        {
            "epoch": 3,
            "model_state_dict": {},
            "config_hash": "different_hash_9999",
        },
        latest_ckpt,
    )

    config = {
        "exp_name": exp_name,
        "output_root": str(output_root),
        "data_root": "data",
        "training": {"epochs": 10, "resume": True},
    }

    with pytest.raises(ValueError, match="Config hash mismatch"):
        train(config)


def test_completed_run_is_skipped(tmp_path: Path) -> None:
    """Verify that train() skips execution if checkpoint epoch >= total_epochs."""
    exp_name = "test_completed_exp"
    output_root = tmp_path / "outputs"
    ckpt_dir = output_root / exp_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "exp_name": exp_name,
        "output_root": str(output_root),
        "data_root": "data",
        "training": {"epochs": 10, "resume": True},
    }
    cfg_hash = compute_config_hash(config)

    latest_ckpt = ckpt_dir / "latest_checkpoint.pth"
    torch.save(
        {
            "epoch": 10,
            "model_state_dict": {},
            "config_hash": cfg_hash,
        },
        latest_ckpt,
    )

    # Calling train should detect completed run and return checkpoint path immediately
    returned_path = train(config)
    assert returned_path == latest_ckpt


def test_smoke_test_uses_separate_folder(tmp_path: Path) -> None:
    """Verify that smoke test routes outputs to <exp_name>_smoke folder."""
    base_config = {
        "exp_name": "test_routing",
        "output_root": str(tmp_path / "outputs"),
        "smoke_test": False,
        "smoke_test_epochs": 1,
        "training": {"epochs": 5, "resume": True},
    }

    # Simulate completed checkpoint in smoke folder
    smoke_dir = tmp_path / "outputs" / "test_routing_smoke" / "checkpoints"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    smoke_config = dict(base_config)
    smoke_config["smoke_test"] = True
    smoke_hash = compute_config_hash(smoke_config)

    smoke_ckpt = smoke_dir / "latest_checkpoint.pth"
    torch.save({"epoch": 1, "config_hash": smoke_hash}, smoke_ckpt)

    res = train(smoke_config)
    assert "test_routing_smoke" in str(res)


def test_unmargined_training_accuracy_logging(tmp_path: Path) -> None:
    """Verify that unmargined training top-1 accuracy is logged to train_metrics.csv."""
    csv_file = tmp_path / "train_metrics.csv"
    logger = CSVMetricLogger(csv_file, resume=False)
    assert "acc" in logger.fieldnames
    assert "train_acc" in logger.fieldnames

    # Test CE linear head
    model_ce = ReIDBackbone(
        backbone_name="resnet18",
        embedding_dim=64,
        num_classes=10,
        classifier_type="linear",
    )
    model_ce.train()
    x = torch.randn(4, 3, 256, 128)
    pids = torch.tensor([0, 1, 2, 3])
    logits_ce, _ = model_ce(x, targets=pids)
    unmargined_ce = model_ce.get_unmargined_logits()
    assert torch.equal(logits_ce, unmargined_ce)

    # Test Margin head (CosFace)
    model_cos = ReIDBackbone(
        backbone_name="resnet18",
        embedding_dim=64,
        num_classes=10,
        classifier_type="cosface",
        margin_s=30.0,
        margin_m=0.35,
    )
    model_cos.train()
    logits_cos, _ = model_cos(x, targets=pids)
    unmargined_cos = model_cos.get_unmargined_logits()
    assert unmargined_cos is not None
    # Margined logits must differ from unmargined logits on target indices
    assert not torch.allclose(logits_cos, unmargined_cos)
    # Target logits must have margin penalty: s * (cos - m) == unmargined - s * m
    target_diff = (
        unmargined_cos[torch.arange(4), pids] - logits_cos[torch.arange(4), pids]
    )
    expected_diff = torch.tensor(30.0 * 0.35)
    assert torch.allclose(target_diff, expected_diff, atol=1e-4)

    # Log an epoch with accuracy
    acc_val = 75.50
    logger.log_epoch({"epoch": 1, "loss": 1.2, "acc": acc_val, "train_acc": acc_val})
    content = csv_file.read_text(encoding="utf-8")
    assert "acc" in content
    assert "75.5" in content
