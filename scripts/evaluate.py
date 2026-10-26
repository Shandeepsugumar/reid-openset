"""Evaluate ReID checkpoint: extract embeddings, compute closed/open-set metrics, and plot ROC."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Person ReID checkpoint.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pth file (defaults to latest checkpoint in exp folder).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to dataset root directory.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Path to output root directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override ('cuda' or 'cpu').",
    )
    return parser.parse_args()


@torch.no_grad()
def extract_split_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Extracting",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    is_cuda = device.type == "cuda"
    embeddings_list, pids_list, camids_list = [], [], []

    for images, pids, camids in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=is_cuda)
        if is_cuda:
            with torch.amp.autocast("cuda", enabled=True):
                feats = model(images)
        else:
            feats = model(images)

        embeddings_list.append(feats.cpu().numpy())
        pids_list.append(pids.numpy())
        camids_list.append(camids.numpy())

    return (
        np.vstack(embeddings_list),
        np.concatenate(pids_list),
        np.concatenate(camids_list),
    )


def evaluate(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_name = config.get("exp_name", "baseline_resnet18")
    raw_output_root = (
        (args.output_root if args else None)
        or os.environ.get("REID_OUTPUT_ROOT")
        or config.get("output_root", "outputs")
    )
    output_root = Path(raw_output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    exp_dir = output_root / exp_name
    embeddings_dir = exp_dir / "embeddings"
    results_dir = exp_dir / "results"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resolve Checkpoint
    if checkpoint_path is None and args and args.checkpoint:
        checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = str(exp_dir / "checkpoints" / "latest_checkpoint.pth")

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

    # Device
    device_name = (args.device if args else None) or config.get("training", {}).get(
        "device", "cuda"
    )
    if device_name.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        device = torch.device("cpu")
    print(f"[Eval] Using device: {device} | Checkpoint: {ckpt_file}")

    # Dataset
    raw_data_root = (
        (args.data_root if args else None)
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

    query_set = Market1501(root_dir=data_folder, split="query", image_size=image_size)
    gallery_set = Market1501(
        root_dir=data_folder, split="gallery", image_size=image_size
    )
    train_set = Market1501(root_dir=data_folder, split="train", image_size=image_size)

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

    # Model
    model_cfg = config.get("model", {})
    model = ReIDBackbone(
        backbone_name=model_cfg.get("backbone", "resnet18"),
        embedding_dim=model_cfg.get("embedding_dim", 512),
        num_classes=train_set.num_classes,
        pretrained=False,
        neck_feat=model_cfg.get("neck_feat", "after"),
    ).to(device)

    ckpt = torch.load(ckpt_file, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # Extract Features
    print("[Eval] Extracting query embeddings...")
    q_feats, q_pids, q_cams = extract_split_features(
        model, query_loader, device, desc="Query Set"
    )
    print("[Eval] Extracting gallery embeddings...")
    g_feats, g_pids, g_cams = extract_split_features(
        model, gallery_loader, device, desc="Gallery Set"
    )

    # Save .npy arrays
    np.save(embeddings_dir / "query_embeddings.npy", q_feats)
    np.save(embeddings_dir / "query_pids.npy", q_pids)
    np.save(embeddings_dir / "query_camids.npy", q_cams)
    np.save(embeddings_dir / "gallery_embeddings.npy", g_feats)
    np.save(embeddings_dir / "gallery_pids.npy", g_pids)
    np.save(embeddings_dir / "gallery_camids.npy", g_cams)
    print(f"[Eval] Embeddings saved to: {embeddings_dir}")

    # Closed-set metrics
    cmc, mAP = compute_closed_set_cmc_and_map(
        q_feats, q_pids, q_cams, g_feats, g_pids, g_cams
    )
    print("\n" + "=" * 45)
    print("  CLOSED-SET RE-IDENTIFICATION RESULTS")
    print("=" * 45)
    print(f"  Rank-1  Accuracy: {cmc[0] * 100:.2f}%")
    print(f"  Rank-5  Accuracy: {cmc[4] * 100:.2f}%")
    print(f"  Rank-10 Accuracy: {cmc[9] * 100:.2f}%")
    print(f"  Mean AP (mAP):   {mAP * 100:.2f}%")
    print("=" * 45)

    # Open-set metrics
    openset_cfg = config.get("openset", {})
    known_ratio = float(openset_cfg.get("known_ratio", 0.5))
    openset_seed = int(openset_cfg.get("seed", 42))

    openset_results = compute_openset_tar_far(
        q_feats,
        q_pids,
        q_cams,
        g_feats,
        g_pids,
        g_cams,
        known_ratio=known_ratio,
        seed=openset_seed,
    )
    print("\n" + "=" * 45)
    print("  OPEN-SET PERSON RE-ID METRICS")
    print("=" * 45)
    print(f"  Enrolled Probes : {openset_results['num_enrolled_probes']}")
    print(f"  Impostor Probes : {openset_results['num_unknown_probes']}")
    print(f"  FAR @ TAR=80%   : {openset_results['far_at_tar_80'] * 100:.2f}%")
    print(f"  FAR @ TAR=90%   : {openset_results['far_at_tar_90'] * 100:.2f}%")
    print(f"  FAR @ TAR=95%   : {openset_results['far_at_tar_95'] * 100:.2f}%")
    print("=" * 45)

    # Save ROC plot
    roc_plot_path = results_dir / "roc_curve.png"
    plt.figure(figsize=(7, 6), dpi=150)
    plt.plot(
        openset_results["far"],
        openset_results["tar"],
        color="#1f77b4",
        lw=2,
        label="Open-Set ReID ROC",
    )
    plt.axvline(
        openset_results["far_at_tar_80"],
        color="gray",
        linestyle="--",
        alpha=0.6,
        label="TAR = 80%",
    )
    plt.axvline(
        openset_results["far_at_tar_90"],
        color="orange",
        linestyle="--",
        alpha=0.6,
        label="TAR = 90%",
    )
    plt.axvline(
        openset_results["far_at_tar_95"],
        color="red",
        linestyle="--",
        alpha=0.6,
        label="TAR = 95%",
    )
    plt.title("Open-Set ReID: Detection & Identification Rate (TAR) vs FAR")
    plt.xlabel("False Alarm Rate (FAR)")
    plt.ylabel("True Accept Rate / Rank-1 Identification (TAR)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_plot_path)
    plt.close()

    # Save results JSON
    results_json = {
        "checkpoint": str(ckpt_file),
        "rank_1": float(cmc[0]),
        "rank_5": float(cmc[4]),
        "rank_10": float(cmc[9]),
        "mAP": float(mAP),
        "far_at_tar_80": openset_results["far_at_tar_80"],
        "far_at_tar_90": openset_results["far_at_tar_90"],
        "far_at_tar_95": openset_results["far_at_tar_95"],
    }
    with open(results_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    return results_json


def main() -> None:
    args = parse_args()
    evaluate(args.config, args.checkpoint, args)


if __name__ == "__main__":
    main()
