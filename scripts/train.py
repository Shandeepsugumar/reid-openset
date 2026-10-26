"""Training script for Open-Set Person ReID with baseline loss and auto-resume."""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reid.data import Market1501, PKSampler  # noqa: E402
from reid.losses import BaselineReIDLoss  # noqa: E402
from reid.models import ReIDBackbone  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Person ReID model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override ('cuda', 'cpu', or 'cuda:0').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override total epochs in config.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to dataset root directory (overrides config and REID_DATA_ROOT).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable automatic checkpoint resumption.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    resolved_path = Path(config_path)
    if not resolved_path.is_file():
        # Try relative to project root
        resolved_path = PROJECT_ROOT / config_path
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class CSVMetricLogger:
    """Logs training metrics to CSV file with flush-on-write."""

    def __init__(self, log_path: Path, resume: bool = False) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = [
            "epoch",
            "lr",
            "loss",
            "loss_ce",
            "loss_triplet",
            "dist_ap",
            "dist_an",
            "epoch_time_sec",
        ]

        file_exists = self.log_path.is_file()
        if not file_exists or not resume:
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_epoch(self, metrics: Dict[str, Any]) -> None:
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            row = {k: metrics.get(k, 0.0) for k in self.fieldnames}
            writer.writerow(row)
            f.flush()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Device configuration
    device_name = args.device or config["training"].get("device", "cuda")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print(f"[Device] CUDA requested ('{device_name}'), but CUDA is unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    print(f"[Device] Using device: {device}")

    # Checkpoint and logging directories
    checkpoint_dir = PROJECT_ROOT / config["training"].get("checkpoint_dir", "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_name = config["training"].get("checkpoint_filename", "latest_checkpoint.pth")
    latest_ckpt_path = checkpoint_dir / latest_ckpt_name

    csv_log_path = PROJECT_ROOT / config["training"].get("csv_log_path", "logs/train_metrics.csv")

    # Dataset & Dataloader
    data_cfg = config.get("data", {})
    raw_dataset_dir = args.data_dir or os.environ.get("REID_DATA_ROOT") or data_cfg.get("dataset_dir", "data")
    dataset_dir = Path(raw_dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_ROOT / dataset_dir
    image_size = tuple(data_cfg.get("image_size", (256, 128)))
    p = data_cfg.get("p", 16)
    k = data_cfg.get("k", 4)
    batch_size = p * k
    num_workers = data_cfg.get("num_workers", 0)

    print(f"[Data] Loading Market-1501 from {dataset_dir}...")
    train_dataset = Market1501(
        root_dir=dataset_dir,
        split="train",
        image_size=image_size,
    )
    print(
        f"[Data] Train set: {len(train_dataset)} images, {train_dataset.num_classes} identities, "
        f"batch size = P({p}) x K({k}) = {batch_size}"
    )

    sampler = PKSampler(train_dataset, p=p, k=k)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Model
    model_cfg = config.get("model", {})
    backbone_name = model_cfg.get("backbone", "resnet18")
    embedding_dim = model_cfg.get("embedding_dim", 512)
    pretrained = model_cfg.get("pretrained", False)
    neck_feat = model_cfg.get("neck_feat", "after")

    model = ReIDBackbone(
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        num_classes=train_dataset.num_classes,
        pretrained=pretrained,
        neck_feat=neck_feat,
    )
    model.to(device)

    # Loss
    loss_cfg = config.get("loss", {})
    criterion = BaselineReIDLoss(
        margin=loss_cfg.get("margin", 0.3),
        label_smoothing=loss_cfg.get("label_smoothing", 0.1),
        weight_ce=loss_cfg.get("weight_ce", 1.0),
        weight_triplet=loss_cfg.get("weight_triplet", 1.0),
        soft_margin=loss_cfg.get("soft_margin", False),
    )
    criterion.to(device)

    # Optimizer
    train_cfg = config.get("training", {})
    lr = float(train_cfg.get("lr", 0.00035))
    weight_decay = float(train_cfg.get("weight_decay", 0.0005))
    total_epochs = args.epochs or int(train_cfg.get("epochs", 60))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Mixed precision setup
    use_amp = bool(train_cfg.get("use_amp", True)) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[Training] Mixed precision (AMP): {'Enabled' if use_amp else 'Disabled'}")

    # Resume from checkpoint if present
    start_epoch = 1
    should_resume = (not args.no_resume) and train_cfg.get("resume", True)
    if should_resume and latest_ckpt_path.is_file():
        print(f"[Resume] Loading latest checkpoint from: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if use_amp and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"[Resume] Successfully resumed. Starting from epoch {start_epoch}/{total_epochs}")
    else:
        if not latest_ckpt_path.is_file():
            print(f"[Checkpoint] No existing checkpoint found at {latest_ckpt_path}. Starting fresh.")
        else:
            print("[Checkpoint] Resumption disabled by flag. Starting fresh.")

    # Metric Logger
    metric_logger = CSVMetricLogger(
        log_path=csv_log_path, resume=(start_epoch > 1)
    )

    if start_epoch > total_epochs:
        print(f"[Training] Model already trained for {start_epoch - 1} epochs (Total epochs: {total_epochs}). Done.")
        return

    # Training loop
    save_epoch_freq = int(train_cfg.get("save_epoch_freq", 1))

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start_time = time.time()
        model.train()

        running_loss = 0.0
        running_ce = 0.0
        running_triplet = 0.0
        running_d_ap = 0.0
        running_d_an = 0.0
        num_batches = 0

        for images, pids, _ in train_loader:
            images = images.to(device, non_blocking=True)
            pids = pids.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, norm_feat = model(images)
                loss, step_metrics = criterion(logits, norm_feat, pids)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += step_metrics["loss"]
            running_ce += step_metrics["loss_ce"]
            running_triplet += step_metrics["loss_triplet"]
            running_d_ap += step_metrics["dist_ap"]
            running_d_an += step_metrics["dist_an"]
            num_batches += 1

        elapsed = time.time() - epoch_start_time
        avg_loss = running_loss / max(1, num_batches)
        avg_ce = running_ce / max(1, num_batches)
        avg_triplet = running_triplet / max(1, num_batches)
        avg_ap = running_d_ap / max(1, num_batches)
        avg_an = running_d_an / max(1, num_batches)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{total_epochs:03d}] "
            f"Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Triplet: {avg_triplet:.4f}) | "
            f"d(ap): {avg_ap:.3f}, d(an): {avg_an:.3f} | "
            f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s"
        )

        epoch_metrics = {
            "epoch": epoch,
            "lr": current_lr,
            "loss": round(avg_loss, 5),
            "loss_ce": round(avg_ce, 5),
            "loss_triplet": round(avg_triplet, 5),
            "dist_ap": round(avg_ap, 4),
            "dist_an": round(avg_an, 4),
            "epoch_time_sec": round(elapsed, 2),
        }
        metric_logger.log_epoch(epoch_metrics)

        # Save checkpoint every epoch (essential for Colab disconnects)
        if epoch % save_epoch_freq == 0 or epoch == total_epochs:
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "config": config,
                "metrics": epoch_metrics,
            }
            # Save latest checkpoint
            torch.save(ckpt_data, latest_ckpt_path)
            # Save periodic epoch checkpoint
            epoch_ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(ckpt_data, epoch_ckpt_path)
            print(f" -> Checkpoint saved: {latest_ckpt_path.name} & {epoch_ckpt_path.name}")

    print(f"[Done] Training completed. Metrics logged to {csv_log_path}")


if __name__ == "__main__":
    main()
