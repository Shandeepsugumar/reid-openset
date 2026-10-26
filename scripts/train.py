"""Training script for Open-Set Person ReID with mixed precision and auto-resume."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reid.data import Market1501, PKSampler  # noqa: E402
from reid.losses import BaselineReIDLoss  # noqa: E402
from reid.models import ReIDBackbone  # noqa: E402


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute a canonical SHA256 hash of configuration dictionary."""
    canonical_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Person ReID model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to dataset root directory (overrides config & REID_DATA_ROOT).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Path to outputs root directory (overrides config & REID_OUTPUT_ROOT).",
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
        "--smoke-test",
        action="store_true",
        help="Run smoke test mode with reduced identities and epochs.",
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
            "gpu_mem_gb",
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


def train(config: Dict[str, Any], args: Optional[argparse.Namespace] = None) -> Path:
    # 1. Resolve configuration and experiment directory
    exp_name = config.get("exp_name", "baseline_resnet18")
    smoke_test = bool(config.get("smoke_test", False))
    if args and args.smoke_test:
        smoke_test = True

    # Dedicated experiment subfolder; smoke test never collides with real run
    exp_folder_name = f"{exp_name}_smoke" if smoke_test else exp_name

    raw_output_root = (
        (args.output_root if args else None)
        or os.environ.get("REID_OUTPUT_ROOT")
        or config.get("output_root", "outputs")
    )
    output_root = Path(raw_output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    exp_dir = output_root / exp_folder_name
    checkpoint_dir = exp_dir / "checkpoints"
    results_dir = exp_dir / "results"
    embeddings_dir = exp_dir / "embeddings"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    csv_log_path = exp_dir / "train_metrics.csv"
    latest_ckpt_path = checkpoint_dir / "latest_checkpoint.pth"

    # Compute configuration hash for resume verification
    cfg_hash = compute_config_hash(config)

    # 2. Seeding & Device selection
    seed = int(config.get("data", {}).get("seed", config.get("seed", 42)))
    set_seed(seed)

    device_override = args.device if args else None
    requested_device = device_override or config.get("training", {}).get("device", "cuda")
    if requested_device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(requested_device)
        is_cuda = True
    else:
        device = torch.device("cpu")
        is_cuda = False
    print(f"[Device] Using device: {device} (CUDA available: {torch.cuda.is_available()})")

    # 3. Epochs and parameters
    train_cfg = config.get("training", {})
    if smoke_test:
        total_epochs = int(config.get("smoke_test_epochs", 2))
        smoke_identities = int(config.get("smoke_test_identities", 50))
    else:
        total_epochs = int(args.epochs) if (args and args.epochs) else int(train_cfg.get("epochs", 60))
        smoke_identities = None

    # Check for completed run
    should_resume = (not (args and args.no_resume)) and train_cfg.get("resume", True)
    if latest_ckpt_path.is_file() and should_resume:
        checkpoint = torch.load(latest_ckpt_path, map_location="cpu")
        saved_epoch = checkpoint.get("epoch", 0)
        saved_hash = checkpoint.get("config_hash")

        # Refuse to resume if config hash differs
        if saved_hash and saved_hash != cfg_hash:
            raise ValueError(
                f"[Resume Error] Config hash mismatch! Checkpoint at {latest_ckpt_path} "
                f"was saved with hash '{saved_hash}', but current config hash is '{cfg_hash}'. "
                "Refusing to resume with mismatched hyperparameters."
            )

        if saved_epoch >= total_epochs:
            print(
                f"[Training] Run '{exp_folder_name}' is already complete "
                f"({saved_epoch}/{total_epochs} epochs). Skipping training."
            )
            return latest_ckpt_path

    # 4. Dataset & DataLoader
    raw_data_root = (
        (args.data_root if args else None)
        or os.environ.get("REID_DATA_ROOT")
        or config.get("data_root", "data")
    )
    dataset_dir = Path(raw_data_root)
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_ROOT / dataset_dir

    data_cfg = config.get("data", {})
    image_size = tuple(data_cfg.get("image_size", (256, 128)))
    p = int(data_cfg.get("p", 16))
    k = int(data_cfg.get("k", 4))
    batch_size = p * k
    num_workers = int(data_cfg.get("num_workers", 2 if is_cuda else 0))

    print(f"[Data] Loading Market-1501 from {dataset_dir} (Smoke test: {smoke_test})...")
    train_dataset = Market1501(
        root_dir=dataset_dir,
        split="train",
        image_size=image_size,
        smoke_test=smoke_test,
        smoke_identities=smoke_identities or 50,
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
        pin_memory=is_cuda,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    # 5. Model Architecture
    model_cfg = config.get("model", {})
    model = ReIDBackbone(
        backbone_name=model_cfg.get("backbone", "resnet18"),
        embedding_dim=model_cfg.get("embedding_dim", 512),
        num_classes=train_dataset.num_classes,
        pretrained=model_cfg.get("pretrained", True),
        neck_feat=model_cfg.get("neck_feat", "after"),
    ).to(device)

    # 6. Loss
    loss_cfg = config.get("loss", {})
    criterion = BaselineReIDLoss(
        margin=float(loss_cfg.get("margin", 0.3)),
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.1)),
        weight_ce=float(loss_cfg.get("weight_ce", 1.0)),
        weight_triplet=float(loss_cfg.get("weight_triplet", 1.0)),
    ).to(device)

    # 7. Optimizer & Warmup+Cosine Scheduler
    lr = float(train_cfg.get("lr", 3.5e-4))
    min_lr = float(train_cfg.get("min_lr", 1.0e-6))
    weight_decay = float(train_cfg.get("weight_decay", 5.0e-4))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 5))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / lr, cosine_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Modern AMP API (no deprecation warnings)
    use_amp = bool(train_cfg.get("use_amp", True)) and is_cuda
    if is_cuda:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = None

    # 8. Resume from checkpoint
    start_epoch = 1
    best_loss = float("inf")
    if latest_ckpt_path.is_file() and should_resume:
        print(f"[Resume] Loading checkpoint from: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"[Resume] Successfully resumed from Epoch {start_epoch}/{total_epochs}")

    # Initialize Metric Logger
    metric_logger = CSVMetricLogger(log_path=csv_log_path, resume=(start_epoch > 1))

    # 9. Training Loop
    print("\n" + "=" * 65)
    print(f"--- STARTING TRAINING: {exp_folder_name} ---")
    print(f"Total Epochs: {total_epochs} | Device: {device} | AMP: {use_amp}")
    print(f"Artifacts: {exp_dir}")
    print("=" * 65)

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()
        model.train()
        running = defaultdict(float)
        steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d}", leave=False)
        for images, pids, _ in pbar:
            images = images.to(device, non_blocking=is_cuda)
            pids = pids.to(device, non_blocking=is_cuda)

            optimizer.zero_grad(set_to_none=True)

            if is_cuda and use_amp:
                with torch.amp.autocast("cuda", enabled=True):
                    logits, norm_feat = model(images)
                    loss, metrics = criterion(logits, norm_feat, pids)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, norm_feat = model(images)
                loss, metrics = criterion(logits, norm_feat, pids)
                loss.backward()
                optimizer.step()

            for k_m, v_m in metrics.items():
                running[k_m] += v_m
            steps += 1
            pbar.set_postfix({"loss": f"{metrics['loss']:.3f}", "ce": f"{metrics['loss_ce']:.3f}"})

        scheduler.step()
        elapsed = time.time() - t0
        gpu_mem = (torch.cuda.max_memory_allocated(device) / (1024**3)) if is_cuda else 0.0
        cur_lr = optimizer.param_groups[0]["lr"]

        avg_loss = running["loss"] / max(1, steps)
        avg_ce = running["loss_ce"] / max(1, steps)
        avg_tri = running["loss_triplet"] / max(1, steps)
        avg_ap = running["dist_ap"] / max(1, steps)
        avg_an = running["dist_an"] / max(1, steps)

        print(
            f"Epoch [{epoch:03d}/{total_epochs:03d}] "
            f"Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Tri: {avg_tri:.4f}) | "
            f"d(ap): {avg_ap:.3f}, d(an): {avg_an:.3f} | "
            f"LR: {cur_lr:.6f} | "
            f"{f'GPU Mem: {gpu_mem:.2f}GB | ' if is_cuda else ''}Time: {elapsed:.1f}s"
        )

        epoch_metrics = {
            "epoch": epoch,
            "lr": cur_lr,
            "loss": round(avg_loss, 5),
            "loss_ce": round(avg_ce, 5),
            "loss_triplet": round(avg_triplet, 5),
            "dist_ap": round(avg_ap, 4),
            "dist_an": round(avg_an, 4),
            "gpu_mem_gb": round(gpu_mem, 3),
            "epoch_time_sec": round(elapsed, 2),
        }
        metric_logger.log_epoch(epoch_metrics)

        # 10. Checkpoint policy:
        # Save latest full state every epoch
        ckpt_full = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "config": config,
            "config_hash": cfg_hash,
            "best_loss": min(best_loss, avg_loss),
            "metrics": epoch_metrics,
        }
        torch.save(ckpt_full, latest_ckpt_path)

        # Save model-only snapshot every 10 epochs and at the end
        if epoch % 10 == 0 or epoch == total_epochs:
            snapshot_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            ckpt_snapshot = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config_hash": cfg_hash,
            }
            torch.save(ckpt_snapshot, snapshot_path)

    print(f"\n[Training] Complete. Latest checkpoint: {latest_ckpt_path}")
    return latest_ckpt_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train(config, args)


if __name__ == "__main__":
    main()
