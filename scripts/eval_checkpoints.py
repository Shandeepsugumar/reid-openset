"""Evaluate checkpoint snapshots to diagnose overfitting/underfitting across training epochs.

NOTE: This diagnostic tool is strictly intended for post-hoc analysis and diagnosing
underfitting or overfitting dynamics over training epochs. It MUST NOT be used for
cherry-picking or tuning hyperparameters/epochs against the test set.
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reid.data import Market1501  # noqa: E402
from reid.eval import (
    compute_closed_set_cmc_and_map,
    compute_openset_tar_far,
)  # noqa: E402
from reid.models import ReIDBackbone  # noqa: E402
from scripts.evaluate import extract_split_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple checkpoints to analyze learning and over/underfitting trends."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing checkpoint_epoch_*.pth snapshots.",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default=None,
        help="Path to train_metrics.csv to plot loss against evaluation curves.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Dataset root path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for CSV and diagnosis plot (defaults to <checkpoint-dir>/../results).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override ('cuda' or 'cpu').",
    )
    return parser.parse_args()


def load_train_loss_history(csv_path: Optional[Path]) -> Dict[int, Dict[str, float]]:
    loss_history = {}
    if csv_path and csv_path.is_file():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    epoch = int(row["epoch"])
                    loss_history[epoch] = {
                        "loss": float(row.get("loss", 0.0)),
                        "loss_ce": float(row.get("loss_ce", 0.0)),
                        "loss_triplet": float(row.get("loss_triplet", 0.0)),
                    }
                except (ValueError, KeyError):
                    continue
    return loss_history


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)
    assert ckpt_dir.is_dir(), f"Checkpoint directory not found: {ckpt_dir}"

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Output directory
    output_dir = (
        Path(args.output_dir) if args.output_dir else (ckpt_dir.parent / "results")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find checkpoints matching checkpoint_epoch_*.pth
    pattern = re.compile(r"checkpoint_epoch_(\d+)\.pth")
    checkpoints = []
    for f in ckpt_dir.glob("checkpoint_epoch_*.pth"):
        m = pattern.search(f.name)
        if m:
            checkpoints.append((int(m.group(1)), f))

    checkpoints.sort(key=lambda x: x[0])
    if not checkpoints:
        print(
            f"[Diagnosis] No checkpoints matching 'checkpoint_epoch_*.pth' found in {ckpt_dir}"
        )
        return

    print(f"[Diagnosis] Found {len(checkpoints)} checkpoint snapshot(s) in {ckpt_dir}")

    # Device
    device_str = args.device or config.get("training", {}).get("device", "cuda")
    device = torch.device(
        device_str
        if torch.cuda.is_available() and device_str.startswith("cuda")
        else "cpu"
    )
    print(f"[Diagnosis] Using device: {device}")

    # Data loaders
    raw_data_root = (
        args.data_root
        or os.environ.get("REID_DATA_ROOT")
        or config.get("data_root", "data")
    )
    data_folder = Path(raw_data_root)
    if not data_folder.is_absolute():
        data_folder = PROJECT_ROOT / data_folder

    image_size = tuple(config.get("data", {}).get("image_size", (256, 128)))
    num_workers = int(
        config.get("data", {}).get("num_workers", 2 if device.type == "cuda" else 0)
    )

    query_set = Market1501(data_folder, split="query", image_size=image_size)
    gallery_set = Market1501(data_folder, split="gallery", image_size=image_size)

    query_loader = DataLoader(
        query_set,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    gallery_loader = DataLoader(
        gallery_set,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Determine num_classes from checkpoint if available
    num_classes = 751
    if checkpoints:
        try:
            first_ckpt = torch.load(checkpoints[0][1], map_location=device)
            first_sd = (
                first_ckpt["model_state_dict"]
                if "model_state_dict" in first_ckpt
                else first_ckpt
            )
            if "classifier.weight" in first_sd:
                num_classes = first_sd["classifier.weight"].size(0)
        except Exception:
            pass

    # Load Model structure
    loss_cls = str(config.get("loss", {}).get("cls", "ce")).lower()
    model = ReIDBackbone(
        backbone_name=config.get("model", {}).get("backbone", "resnet18"),
        embedding_dim=config.get("model", {}).get("embedding_dim", 512),
        num_classes=num_classes,
        pretrained=False,
        neck_feat=config.get("model", {}).get("neck_feat", "after"),
        classifier_type=loss_cls,
    ).to(device)

    # Load train loss history if provided
    train_csv_path = (
        Path(args.train_csv)
        if args.train_csv
        else (ckpt_dir.parent / "train_metrics.csv")
    )
    train_history = load_train_loss_history(train_csv_path)

    # Evaluate each snapshot
    known_ratio = float(config.get("openset", {}).get("known_ratio", 0.5))
    openset_seed = int(config.get("openset", {}).get("seed", 42))

    eval_rows = []
    for epoch, ckpt_path in checkpoints:
        print(f"\n[Diagnosis] Evaluating Epoch {epoch:03d} ({ckpt_path.name})...")
        ckpt_data = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt_data.get("model_state_dict", ckpt_data)
        model.load_state_dict(state_dict)
        model.eval()

        q_feats, q_pids, q_cams = extract_split_features(
            model, query_loader, device, desc=f"Query Ep {epoch}"
        )
        g_feats, g_pids, g_cams = extract_split_features(
            model, gallery_loader, device, desc=f"Gallery Ep {epoch}"
        )

        cmc, mAP = compute_closed_set_cmc_and_map(
            q_feats, q_pids, q_cams, g_feats, g_pids, g_cams
        )
        openset_res = compute_openset_tar_far(
            q_feats,
            q_pids,
            q_cams,
            g_feats,
            g_pids,
            g_cams,
            known_ratio=known_ratio,
            seed=openset_seed,
        )

        loss_entry = train_history.get(epoch, {})
        row = {
            "epoch": epoch,
            "train_loss": loss_entry.get("loss", "N/A"),
            "train_loss_ce": loss_entry.get("loss_ce", "N/A"),
            "train_loss_triplet": loss_entry.get("loss_triplet", "N/A"),
            "rank_1": round(float(cmc[0] * 100), 2),
            "rank_5": round(float(cmc[4] * 100), 2),
            "mAP": round(float(mAP * 100), 2),
            "far_at_tar_80": round(float(openset_res["far_at_tar_80"] * 100), 2),
            "far_at_tar_90": round(float(openset_res["far_at_tar_90"] * 100), 2),
        }
        eval_rows.append(row)
        print(
            f"  -> Rank-1: {row['rank_1']:.2f}% | mAP: {row['mAP']:.2f}% | "
            f"FAR@80: {row['far_at_tar_80']:.2f}% | FAR@90: {row['far_at_tar_90']:.2f}%"
        )

    # Save to CSV
    csv_out_path = output_dir / "checkpoint_eval_metrics.csv"
    with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "epoch",
            "train_loss",
            "train_loss_ce",
            "train_loss_triplet",
            "rank_1",
            "rank_5",
            "mAP",
            "far_at_tar_80",
            "far_at_tar_90",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eval_rows)
    print(f"\n[Diagnosis] Metrics saved to: {csv_out_path}")

    # Plot diagnosis curve
    plot_path = output_dir / "checkpoint_diagnosis_curve.png"
    epochs = [r["epoch"] for r in eval_rows]
    r1s = [r["rank_1"] for r in eval_rows]
    maps = [r["mAP"] for r in eval_rows]
    far80s = [r["far_at_tar_80"] for r in eval_rows]
    far90s = [r["far_at_tar_90"] for r in eval_rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), dpi=150, sharex=True)

    # Plot loss curve if train metrics are available
    if train_history:
        sorted_epochs = sorted(train_history.keys())
        losses = [train_history[e]["loss"] for e in sorted_epochs]
        ax1.plot(
            sorted_epochs, losses, color="purple", label="Total Training Loss", lw=2
        )
        ax1.set_ylabel("Training Loss")
        ax1.set_title("Training Loss Dynamics across Epochs")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right")
    else:
        ax1.text(0.5, 0.5, "Train loss CSV not found", ha="center", va="center")

    # Plot test metrics
    ax2.plot(epochs, r1s, marker="o", label="Rank-1 (%)", color="#1f77b4")
    ax2.plot(epochs, maps, marker="s", label="mAP (%)", color="#2ca02c")
    ax2.plot(
        epochs,
        far80s,
        marker="^",
        label="FAR @ TAR=80% (%)",
        color="#ff7f0e",
        linestyle="--",
    )
    ax2.plot(
        epochs,
        far90s,
        marker="v",
        label="FAR @ TAR=90% (%)",
        color="#d62728",
        linestyle="--",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric (%)")
    ax2.set_title("Evaluation Performance Trends (Diagnostic Only)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[Diagnosis] Diagnosis plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
