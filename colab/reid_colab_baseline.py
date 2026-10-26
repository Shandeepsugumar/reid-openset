# %% [markdown]
# # Open-Set Person Re-Identification on Market-1501 (Google Colab T4)
# Unified training, mixed-precision auto-resumption, and open-set evaluation pipeline.

# %% [Cell 1: Setup & Unzip]
# ==============================================================================
# CELL 1: SETUP (Drive Mount, GPU Verification, Dataset Unzip, Sanity Check)
# ==============================================================================
import os
import re
import sys
import math
import json
import time
import shutil
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Sequence
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode

# 1. Mount Google Drive
try:
    from google.colab import drive
    drive.mount("/content/drive")
    print("[Setup] Google Drive mounted successfully at /content/drive")
except ImportError:
    print("[Setup] Not running inside Google Colab. Continuing with local filesystem.")

# 2. Strict GPU Check
assert torch.cuda.is_available(), (
    "\n" + "=" * 70 + "\n"
    "ERROR: GPU runtime not detected!\n"
    "Please switch to a GPU runtime before continuing:\n"
    "  1. Go to top menu: Runtime -> Change runtime type\n"
    "  2. Under 'Hardware accelerator', select 'T4 GPU'\n"
    "  3. Click 'Save' and re-run this notebook.\n" + "=" * 70
)
device = torch.device("cuda")
print(f"[Setup] Active GPU: {torch.cuda.get_device_name(0)}")
os.system("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader")

# 3. Explicit Dataset Extraction from /content/drive/MyDrive/dataset/archive (1).zip
DATASET_ZIP_PATH = Path("/content/drive/MyDrive/dataset/archive (1).zip")
DATA_EXTRACT_DIR = Path("/content/data")
DATA_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

# Check if dataset is already extracted
existing_train_dirs = list(DATA_EXTRACT_DIR.rglob("bounding_box_train"))
if existing_train_dirs:
    RESOLVED_DATASET_ROOT = existing_train_dirs[0].parent
    print(f"[Setup] Dataset already extracted at: {RESOLVED_DATASET_ROOT}. Skipping unzip.")
else:
    # Verify zip file exists
    assert DATASET_ZIP_PATH.is_file(), (
        f"\n[Error] Dataset archive not found at: '{DATASET_ZIP_PATH}'\n"
        f"Please verify that 'archive (1).zip' exists inside your Google Drive under the 'dataset' folder."
    )

    print(f"[Setup] Found dataset archive at: {DATASET_ZIP_PATH}")
    print(f"[Setup] Extracting into {DATA_EXTRACT_DIR} ...")
    with zipfile.ZipFile(DATASET_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_EXTRACT_DIR)
    print("[Setup] Extraction completed successfully.")

    # Locate the folder containing bounding_box_train
    found_train = list(DATA_EXTRACT_DIR.rglob("bounding_box_train"))
    assert len(found_train) > 0, "Extraction finished but 'bounding_box_train' was not found."
    RESOLVED_DATASET_ROOT = found_train[0].parent
    print(f"[Setup] Resolved dataset root: {RESOLVED_DATASET_ROOT}")

# 4. Strict Dataset Sanity Check
def run_dataset_sanity_check(root: Path):
    splits = {
        "train": root / "bounding_box_train",
        "query": root / "query",
        "gallery": root / "bounding_box_test",
    }
    pattern = re.compile(r"([-\d]+)_c(\d+)")
    stats = {}
    print("\n[Setup] Verifying split files:")
    for name, path in splits.items():
        assert path.is_dir(), f"Missing split directory: {path}"
        files = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        pids = set()
        cams = set()
        for f in files:
            m = pattern.match(f)
            if m:
                pids.add(int(m.group(1)))
                cams.add(int(m.group(2)))
        stats[name] = {"files": len(files), "pids": len(pids), "cams": len(cams)}
        print(f"  - {name:<7}: {len(files):>5} images | {len(pids):>4} identities | {len(cams):>2} cameras")

    assert stats["train"]["files"] > 10000, f"Train count unexpected: {stats['train']['files']}"
    assert stats["query"]["files"] > 2000, f"Query count unexpected: {stats['query']['files']}"
    assert stats["gallery"]["files"] > 15000, f"Gallery count unexpected: {stats['gallery']['files']}"
    print("[Setup] Dataset sanity check: PASSED.")

run_dataset_sanity_check(RESOLVED_DATASET_ROOT)


# %% [Cell 2: Config]
# ==============================================================================
# CELL 2: CONFIGURATION
# ==============================================================================
DRIVE_DATA_ZIP = "/content/drive/MyDrive/dataset/archive (1).zip"
CKPT_DIR = "/content/drive/MyDrive/reid_openset/checkpoints"
OUTPUT_DIR = "/content/drive/MyDrive/reid_openset/outputs"
DATA_ROOT = str(RESOLVED_DATASET_ROOT)

CONFIG = {
    # Paths
    "drive_data_zip": DRIVE_DATA_ZIP,
    "ckpt_dir": CKPT_DIR,
    "output_dir": OUTPUT_DIR,
    "data_root": DATA_ROOT,

    # Verification switch: True executes 50 identities & 2 epochs for quick check
    "smoke_test": False,
    "smoke_test_identities": 50,
    "smoke_test_epochs": 2,

    # Architecture
    "backbone": "resnet18",
    "embedding_dim": 512,
    "pretrained": True,
    "neck_feat": "after",  # L2-normalize features 'after' BNNeck

    # Data & Sampling
    "image_size": (256, 128),  # (height, width)
    "p": 16,                   # Number of identities per batch
    "k": 4,                    # Images per identity -> Batch size = P * K = 64
    "num_workers": 2,          # T4 sweet spot in Colab
    "seed": 42,

    # Optimization & Loss
    "epochs": 60,
    "warmup_epochs": 5,
    "lr": 3.5e-4,
    "min_lr": 1.0e-6,
    "weight_decay": 5.0e-4,
    "margin": 0.3,
    "label_smoothing": 0.1,
    "weight_ce": 1.0,
    "weight_triplet": 1.0,

    # Open-Set Protocol
    "openset_known_ratio": 0.5,  # 50% query IDs enrolled, 50% unseen impostors
    "openset_seed": 42,
}

# Create output and checkpoint directories
os.makedirs(CONFIG["ckpt_dir"], exist_ok=True)
os.makedirs(CONFIG["output_dir"], exist_ok=True)
print("[Config] Configuration initialized successfully.")


# %% [Cell 3: Components]
# ==============================================================================
# CELL 3: DATASET, SAMPLER, MODEL, LOSSES
# ==============================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Transforms ---
def build_transforms(
    image_size: Tuple[int, int] = (256, 128),
    is_train: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    random_erase_prob: float = 0.5,
) -> transforms.Compose:
    h, w = image_size
    if is_train:
        t = [
            transforms.Resize((h, w), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        if random_erase_prob > 0.0:
            t.append(transforms.RandomErasing(p=random_erase_prob, value="random"))
        return transforms.Compose(t)
    return transforms.Compose([
        transforms.Resize((h, w), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# --- Dataset ---
class Market1501Dataset(Dataset):
    FILENAME_PATTERN = re.compile(r"([-\d]+)_c(\d+)")

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        image_size: Tuple[int, int] = (256, 128),
        smoke_test: bool = False,
        smoke_identities: int = 50,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = "gallery" if split == "test" else split
        assert self.split in {"train", "query", "gallery"}

        self.transform = build_transforms(image_size=image_size, is_train=(self.split == "train"))
        folder = {
            "train": "bounding_box_train",
            "query": "query",
            "gallery": "bounding_box_test",
        }[self.split]
        folder_path = self.root_dir / folder

        raw_samples = []
        pids_set = set()
        cams_set = set()

        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            m = self.FILENAME_PATTERN.match(fname)
            if not m:
                continue
            pid = int(m.group(1))
            camid = int(m.group(2))

            # Discard junk (-1) and distractor (0) during training
            if self.split == "train" and (pid == -1 or pid == 0):
                continue

            raw_samples.append((str(folder_path / fname), pid, camid))
            pids_set.add(pid)
            cams_set.add(camid)

        # Relabel pids contiguously for classification head during training
        if self.split == "train":
            sorted_pids = sorted(list(pids_set))
            if smoke_test:
                sorted_pids = sorted_pids[:smoke_identities]
                valid_pids = set(sorted_pids)
                raw_samples = [s for s in raw_samples if s[1] in valid_pids]

            self.pid_to_label = {pid: idx for idx, pid in enumerate(sorted_pids)}
            self.label_to_pid = {idx: pid for pid, idx in self.pid_to_label.items()}
            self.samples = [
                (path, self.pid_to_label[pid], camid)
                for path, pid, camid in raw_samples
            ]
            self.num_classes = len(sorted_pids)
        else:
            self.pid_to_label = {pid: pid for pid in pids_set}
            self.label_to_pid = {pid: pid for pid in pids_set}
            self.samples = raw_samples
            self.num_classes = len(pids_set)

        self.num_cameras = len(cams_set)

    @property
    def pids(self) -> List[int]:
        return [s[1] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        path, pid, camid = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, pid, camid


# --- PK Sampler ---
class PKSampler(Sampler[int]):
    def __init__(
        self,
        data_source: Union[Dataset, Sequence[int]],
        p: int = 16,
        k: int = 4,
    ) -> None:
        self.p = p
        self.k = k
        self.batch_size = p * k

        if hasattr(data_source, "pids"):
            self.labels = list(data_source.pids)
        else:
            self.labels = list(data_source)

        self.index_pid_map = defaultdict(list)
        for idx, pid in enumerate(self.labels):
            self.index_pid_map[pid].append(idx)

        self.unique_pids = list(self.index_pid_map.keys())
        assert len(self.unique_pids) >= self.p, (
            f"Unique identities ({len(self.unique_pids)}) < batch P ({self.p})"
        )
        self.num_batches = max(1, len(self.labels) // self.batch_size)
        self.length = self.num_batches * self.batch_size

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        batch_indices = []
        for _ in range(self.num_batches):
            selected_pids = random.sample(self.unique_pids, self.p)
            for pid in selected_pids:
                indices = self.index_pid_map[pid]
                if len(indices) >= self.k:
                    sampled = random.sample(indices, self.k)
                else:
                    sampled = np.random.choice(indices, size=self.k, replace=True).tolist()
                batch_indices.extend(sampled)
        return iter(batch_indices)


# --- Model ---
def weights_init_kaiming(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class ReIDBackbone(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        embedding_dim: int = 512,
        num_classes: Optional[int] = None,
        pretrained: bool = True,
        neck_feat: str = "after",
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.neck_feat = neck_feat

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        in_features = base.fc.in_features
        self.conv_base = nn.Sequential(*list(base.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        if in_features != embedding_dim:
            self.head = nn.Linear(in_features, embedding_dim, bias=False)
            self.head.apply(weights_init_kaiming)
        else:
            self.head = nn.Identity()

        self.bnneck = nn.BatchNorm1d(embedding_dim)
        self.bnneck.bias.requires_grad_(False)
        self.bnneck.apply(weights_init_kaiming)

        if num_classes is not None and num_classes > 0:
            self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
        else:
            self.classifier = None

    def forward(self, x: torch.Tensor):
        x = self.conv_base(x)
        feat = self.head(torch.flatten(self.gap(x), 1))
        feat_bn = self.bnneck(feat)

        target_feat = feat_bn if self.neck_feat == "after" else feat
        norm_feat = F.normalize(target_feat, p=2, dim=1)

        if self.training and self.classifier is not None:
            logits = self.classifier(feat_bn)
            return logits, norm_feat
        return norm_feat


# --- Losses ---
def euclidean_distance_matrix(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    dot_prod = torch.matmul(x, x.t())
    sq_norm = torch.sum(x**2, dim=1, keepdim=True)
    dist_sq = sq_norm - 2.0 * dot_prod + sq_norm.t()
    return torch.sqrt(torch.clamp(dist_sq, min=0.0) + eps)


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor):
        dist_mat = euclidean_distance_matrix(embeddings)
        targets_col = targets.unsqueeze(1)
        targets_row = targets.unsqueeze(0)
        mask_same = torch.eq(targets_col, targets_row)
        mask_diff = torch.ne(targets_col, targets_row)

        eye = torch.eye(targets.size(0), dtype=torch.bool, device=targets.device)
        mask_pos = mask_same & (~eye)

        dist_mat_pos = dist_mat.clone()
        dist_mat_pos[~mask_pos] = -1.0
        dist_ap, _ = torch.max(dist_mat_pos, dim=1)

        dist_mat_neg = dist_mat.clone()
        dist_mat_neg[~mask_diff] = float("inf")
        dist_an, _ = torch.min(dist_mat_neg, dim=1)

        loss = F.relu(dist_ap - dist_an + self.margin).mean()
        return loss, dist_ap, dist_an


class BaselineReIDLoss(nn.Module):
    def __init__(
        self,
        margin: float = 0.3,
        label_smoothing: float = 0.1,
        weight_ce: float = 1.0,
        weight_triplet: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_triplet = weight_triplet
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.triplet = BatchHardTripletLoss(margin=margin)

    def forward(self, logits: torch.Tensor, embeddings: torch.Tensor, targets: torch.Tensor):
        loss_ce = self.ce(logits, targets)
        loss_tri, d_ap, d_an = self.triplet(embeddings, targets)
        total_loss = self.weight_ce * loss_ce + self.weight_triplet * loss_tri
        metrics = {
            "loss": float(total_loss.item()),
            "loss_ce": float(loss_ce.item()),
            "loss_triplet": float(loss_tri.item()),
            "dist_ap": float(d_ap.mean().item()),
            "dist_an": float(d_an.mean().item()),
        }
        return total_loss, metrics


# %% [Cell 4: Training Loop]
# ==============================================================================
# CELL 4: TRAINING FUNCTION WITH AUTO-RESUME & MIXED PRECISION
# ==============================================================================
def train_reid(config: Dict) -> Path:
    set_seed(config["seed"])
    device = torch.device("cuda")

    total_epochs = config["smoke_test_epochs"] if config["smoke_test"] else config["epochs"]
    ckpt_dir = Path(config["ckpt_dir"])
    latest_ckpt_path = ckpt_dir / "latest_checkpoint.pth"
    csv_log_path = Path(config["output_dir"]) / "train_metrics.csv"

    print("\n" + "=" * 65)
    print(f"--- STARTING TRAINING (Smoke Test: {config['smoke_test']}) ---")
    print(f"Total Epochs: {total_epochs} | Device: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"CSV Logs:    {csv_log_path}")
    print("=" * 65)

    # 1. Dataset & DataLoader
    train_dataset = Market1501Dataset(
        root_dir=Path(config["data_root"]),
        split="train",
        image_size=config["image_size"],
        smoke_test=config["smoke_test"],
        smoke_identities=config["smoke_test_identities"],
    )
    sampler = PKSampler(train_dataset, p=config["p"], k=config["k"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["p"] * config["k"],
        sampler=sampler,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=(config["num_workers"] > 0),
        drop_last=True,
    )
    print(f"[Data] Training samples: {len(train_dataset)} across {train_dataset.num_classes} identities")

    # 2. Model & Loss
    model = ReIDBackbone(
        backbone_name=config["backbone"],
        embedding_dim=config["embedding_dim"],
        num_classes=train_dataset.num_classes,
        pretrained=config["pretrained"],
        neck_feat=config["neck_feat"],
    ).to(device)

    criterion = BaselineReIDLoss(
        margin=config["margin"],
        label_smoothing=config["label_smoothing"],
        weight_ce=config["weight_ce"],
        weight_triplet=config["weight_triplet"],
    ).to(device)

    # 3. Optimizer & Warmup-Cosine Scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    warmup_epochs = config["warmup_epochs"]
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(config["min_lr"] / config["lr"], cosine_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # 4. Auto-Resumption Logic
    start_epoch = 1
    best_loss = float("inf")
    if latest_ckpt_path.is_file():
        print(f"\n[Resume] Found checkpoint at '{latest_ckpt_path}'. Resuming...")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"[Resume] Successfully resumed from Epoch {start_epoch}/{total_epochs}.\n")
    else:
        print("[Resume] No existing checkpoint found. Starting fresh from scratch.\n")

    # CSV Header setup
    csv_header = "epoch,lr,loss,loss_ce,loss_triplet,dist_ap,dist_an,gpu_mem_gb,epoch_time_sec\n"
    if start_epoch == 1 or not csv_log_path.is_file():
        with open(csv_log_path, "w", encoding="utf-8") as f:
            f.write(csv_header)

    # 5. Training Loop
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()
        model.train()
        running = defaultdict(float)
        steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d}", leave=False)
        for images, pids, _ in pbar:
            images = images.to(device, non_blocking=True)
            pids = pids.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                logits, norm_feat = model(images)
                loss, metrics = criterion(logits, norm_feat, pids)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            for k_m, v_m in metrics.items():
                running[k_m] += v_m
            steps += 1
            pbar.set_postfix({"loss": f"{metrics['loss']:.3f}", "ce": f"{metrics['loss_ce']:.3f}"})

        scheduler.step()
        elapsed = time.time() - t0
        gpu_mem = torch.cuda.max_memory_allocated(device) / (1024**3)
        cur_lr = optimizer.param_groups[0]["lr"]

        avg_loss = running["loss"] / steps
        avg_ce = running["loss_ce"] / steps
        avg_tri = running["loss_triplet"] / steps
        avg_ap = running["dist_ap"] / steps
        avg_an = running["dist_an"] / steps

        print(
            f"Epoch [{epoch:03d}/{total_epochs:03d}] "
            f"Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Tri: {avg_tri:.4f}) | "
            f"d(ap): {avg_ap:.3f}, d(an): {avg_an:.3f} | "
            f"LR: {cur_lr:.6f} | GPU Mem: {gpu_mem:.2f}GB | Time: {elapsed:.1f}s"
        )

        # Log metrics to CSV
        log_line = (
            f"{epoch},{cur_lr:.6e},{avg_loss:.5f},{avg_ce:.5f},"
            f"{avg_tri:.5f},{avg_ap:.4f},{avg_an:.4f},{gpu_mem:.3f},{elapsed:.2f}\n"
        )
        with open(csv_log_path, "a", encoding="utf-8") as f:
            f.write(log_line)

        # Save Checkpoint EVERY epoch to Drive
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_loss": min(best_loss, avg_loss),
            "config": config,
        }
        torch.save(ckpt_data, latest_ckpt_path)
        torch.save(ckpt_data, ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pth")

    print(f"\n[Training] Complete. Checkpoint saved at: {latest_ckpt_path}")
    return latest_ckpt_path


# %% [Cell 5: Evaluation]
# ==============================================================================
# CELL 5: EMBEDDING EXTRACTION, CLOSED & OPEN-SET EVALUATION
# ==============================================================================
@torch.no_grad()
def extract_split_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Extracting",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    embeddings_list, pids_list, camids_list = [], [], []

    for images, pids, camids in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=True):
            feats = model(images)  # Returns L2-normalized 512-dim features
        embeddings_list.append(feats.cpu().numpy())
        pids_list.append(pids.numpy())
        camids_list.append(camids.numpy())

    return (
        np.vstack(embeddings_list),
        np.concatenate(pids_list),
        np.concatenate(camids_list),
    )


def compute_closed_set_cmc_and_map(
    q_feats: np.ndarray,
    q_pids: np.ndarray,
    q_cams: np.ndarray,
    g_feats: np.ndarray,
    g_pids: np.ndarray,
    g_cams: np.ndarray,
    max_rank: int = 20,
) -> Tuple[np.ndarray, float]:
    """Standard Market-1501 evaluation: ignore same identity in same camera & junk (-1)."""
    sim_mat = np.dot(q_feats, g_feats.T)
    num_queries = len(q_pids)

    all_cmc = []
    all_ap = []

    for i in range(num_queries):
        pid = q_pids[i]
        cam = q_cams[i]

        order = np.argsort(-sim_mat[i])
        valid_mask = ~((g_pids[order] == pid) & (g_cams[order] == cam)) & (g_pids[order] != -1)
        order = order[valid_mask]

        matches = (g_pids[order] == pid).astype(np.int32)
        total_positives = np.sum(matches)
        if total_positives == 0:
            continue

        cmc = np.zeros(max_rank)
        first_match_idx = np.where(matches == 1)[0]
        if len(first_match_idx) > 0 and first_match_idx[0] < max_rank:
            cmc[first_match_idx[0]:] = 1.0
        all_cmc.append(cmc)

        cum_matches = np.cumsum(matches)
        ranks = np.arange(1, len(matches) + 1)
        precisions = cum_matches / ranks
        ap = np.sum(precisions * matches) / total_positives
        all_ap.append(ap)

    mean_cmc = np.mean(all_cmc, axis=0)
    mAP = float(np.mean(all_ap))
    return mean_cmc, mAP


def compute_openset_tar_far(
    q_feats: np.ndarray,
    q_pids: np.ndarray,
    q_cams: np.ndarray,
    g_feats: np.ndarray,
    g_pids: np.ndarray,
    g_cams: np.ndarray,
    known_ratio: float = 0.5,
    seed: int = 42,
) -> Dict:
    """Open-Set ReID: Partition query identities into Enrolled (known) vs Unknown (impostor)."""
    np.random.seed(seed)
    unique_q_pids = np.unique(q_pids)
    np.random.shuffle(unique_q_pids)

    n_known = int(len(unique_q_pids) * known_ratio)
    enrolled_pids = set(unique_q_pids[:n_known])
    unknown_pids = set(unique_q_pids[n_known:])

    gallery_mask = np.array([pid not in unknown_pids for pid in g_pids])
    sub_g_feats = g_feats[gallery_mask]
    sub_g_pids = g_pids[gallery_mask]
    sub_g_cams = g_cams[gallery_mask]

    sim_mat = np.dot(q_feats, sub_g_feats.T)

    target_scores = []
    target_correct_rank1 = []
    impostor_scores = []

    for i in range(len(q_pids)):
        pid = q_pids[i]
        cam = q_cams[i]

        valid_mask = ~((sub_g_pids == pid) & (sub_g_cams == cam)) & (sub_g_pids != -1)
        if not np.any(valid_mask):
            continue

        valid_sims = sim_mat[i][valid_mask]
        valid_pids = sub_g_pids[valid_mask]
        best_idx = np.argmax(valid_sims)
        max_score = valid_sims[best_idx]
        is_rank1 = (valid_pids[best_idx] == pid)

        if pid in enrolled_pids:
            target_scores.append(max_score)
            target_correct_rank1.append(is_rank1)
        elif pid in unknown_pids:
            impostor_scores.append(max_score)

    target_scores = np.array(target_scores)
    target_correct_rank1 = np.array(target_correct_rank1)
    impostor_scores = np.array(impostor_scores)

    thresholds = np.linspace(-0.2, 1.0, 500)
    far_list, tar_list = [], []

    for tau in thresholds:
        far = np.mean(impostor_scores >= tau)
        tar = np.mean((target_scores >= tau) & target_correct_rank1)
        far_list.append(far)
        tar_list.append(tar)

    far_arr = np.array(far_list)
    tar_arr = np.array(tar_list)

    def get_far_at_tar(desired_tar: float) -> float:
        valid_indices = np.where(tar_arr >= desired_tar)[0]
        if len(valid_indices) == 0:
            return 1.0
        return float(np.min(far_arr[valid_indices]))

    return {
        "thresholds": thresholds.tolist(),
        "far": far_arr.tolist(),
        "tar": tar_arr.tolist(),
        "far_at_tar_80": get_far_at_tar(0.80),
        "far_at_tar_90": get_far_at_tar(0.90),
        "far_at_tar_95": get_far_at_tar(0.95),
        "num_enrolled_probes": len(target_scores),
        "num_unknown_probes": len(impostor_scores),
    }


def evaluate_pipeline(config: Dict, checkpoint_path: Optional[Path] = None):
    device = torch.device("cuda")
    data_folder = Path(config["data_root"])
    output_dir = Path(config["output_dir"])

    print("\n" + "=" * 65)
    print("--- STARTING OPEN-SET & CLOSED-SET EVALUATION ---")
    print("=" * 65)

    # 1. Load Data
    query_set = Market1501Dataset(data_folder, split="query", image_size=config["image_size"])
    gallery_set = Market1501Dataset(data_folder, split="gallery", image_size=config["image_size"])

    query_loader = DataLoader(
        query_set, batch_size=128, shuffle=False, num_workers=config["num_workers"], pin_memory=True
    )
    gallery_loader = DataLoader(
        gallery_set, batch_size=128, shuffle=False, num_workers=config["num_workers"], pin_memory=True
    )

    # 2. Load Model
    if checkpoint_path is None or not Path(checkpoint_path).is_file():
        checkpoint_path = Path(config["ckpt_dir"]) / "latest_checkpoint.pth"

    assert checkpoint_path.is_file(), f"Checkpoint not found at: {checkpoint_path}"
    print(f"[Eval] Loading weights from: {checkpoint_path}")

    train_dataset = Market1501Dataset(
        data_folder, split="train", smoke_test=config["smoke_test"], smoke_identities=config["smoke_test_identities"]
    )
    model = ReIDBackbone(
        backbone_name=config["backbone"],
        embedding_dim=config["embedding_dim"],
        num_classes=train_dataset.num_classes,
        pretrained=False,
        neck_feat=config["neck_feat"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. Extract Embeddings
    print("\n[Eval] Extracting query embeddings...")
    q_feats, q_pids, q_cams = extract_split_features(model, query_loader, device, desc="Query Set")
    print(f"       Extracted {q_feats.shape[0]} query embeddings ({q_feats.shape[1]}-dim)")

    print("[Eval] Extracting gallery embeddings...")
    g_feats, g_pids, g_cams = extract_split_features(model, gallery_loader, device, desc="Gallery Set")
    print(f"       Extracted {g_feats.shape[0]} gallery embeddings ({g_feats.shape[1]}-dim)")

    # 4. Save Embeddings to Google Drive
    print(f"\n[Eval] Saving .npy embedding artifacts to: {output_dir}")
    np.save(output_dir / "query_embeddings.npy", q_feats)
    np.save(output_dir / "query_pids.npy", q_pids)
    np.save(output_dir / "query_camids.npy", q_cams)
    np.save(output_dir / "gallery_embeddings.npy", g_feats)
    np.save(output_dir / "gallery_pids.npy", g_pids)
    np.save(output_dir / "gallery_camids.npy", g_cams)
    print("       Successfully saved query and gallery .npy files.")

    # 5. Closed-Set Metrics (Market-1501 Standard)
    print("\n[Eval] Computing closed-set Rank-k and mAP metrics...")
    cmc, mAP = compute_closed_set_cmc_and_map(q_feats, q_pids, q_cams, g_feats, g_pids, g_cams)

    print("\n" + "-" * 40)
    print("  CLOSED-SET RE-IDENTIFICATION RESULTS")
    print("-" * 40)
    print(f"  Rank-1  Accuracy: {cmc[0] * 100:.2f}%")
    print(f"  Rank-5  Accuracy: {cmc[4] * 100:.2f}%")
    print(f"  Rank-10 Accuracy: {cmc[9] * 100:.2f}%")
    print(f"  Mean AP (mAP):   {mAP * 100:.2f}%")
    print("-" * 40)

    # 6. Open-Set Metrics (Enrolled vs Impostor Detection)
    print("\n[Eval] Computing open-set verification metrics (DIR / TAR vs FAR)...")
    openset_results = compute_openset_tar_far(
        q_feats,
        q_pids,
        q_cams,
        g_feats,
        g_pids,
        g_cams,
        known_ratio=config["openset_known_ratio"],
        seed=config["openset_seed"],
    )

    print("\n" + "-" * 40)
    print("  OPEN-SET PERSON RE-ID METRICS")
    print("-" * 40)
    print(f"  Enrolled Probes : {openset_results['num_enrolled_probes']}")
    print(f"  Impostor Probes : {openset_results['num_unknown_probes']}")
    print(f"  FAR @ TAR=80%   : {openset_results['far_at_tar_80'] * 100:.2f}%")
    print(f"  FAR @ TAR=90%   : {openset_results['far_at_tar_90'] * 100:.2f}%")
    print(f"  FAR @ TAR=95%   : {openset_results['far_at_tar_95'] * 100:.2f}%")
    print("-" * 40)

    # 7. Plot and Save ROC Curve
    roc_plot_path = output_dir / "roc_curve.png"
    plt.figure(figsize=(7, 6), dpi=150)
    plt.plot(openset_results["far"], openset_results["tar"], color="#1f77b4", lw=2, label="Open-Set ReID ROC")
    plt.axvline(openset_results["far_at_tar_80"], color="gray", linestyle="--", alpha=0.6, label="TAR = 80%")
    plt.axvline(openset_results["far_at_tar_90"], color="orange", linestyle="--", alpha=0.6, label="TAR = 90%")
    plt.axvline(openset_results["far_at_tar_95"], color="red", linestyle="--", alpha=0.6, label="TAR = 95%")
    plt.title("Open-Set ReID: Detection & Identification Rate (TAR) vs FAR")
    plt.xlabel("False Alarm Rate (FAR)")
    plt.ylabel("True Accept Rate / Rank-1 Identification (TAR)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_plot_path)
    plt.close()
    print(f"[Eval] ROC plot saved to: {roc_plot_path}")

    # 8. Save Metrics JSON
    results_json_path = output_dir / "eval_results.json"
    final_results = {
        "checkpoint": str(checkpoint_path),
        "rank_1": float(cmc[0]),
        "rank_5": float(cmc[4]),
        "rank_10": float(cmc[9]),
        "mAP": float(mAP),
        "far_at_tar_80": openset_results["far_at_tar_80"],
        "far_at_tar_90": openset_results["far_at_tar_90"],
        "far_at_tar_95": openset_results["far_at_tar_95"],
    }
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"[Eval] Full results JSON saved to: {results_json_path}")
    print("\n" + "=" * 65 + "\n[Eval] PIPELINE FINISHED SUCCESSFULLY.\n" + "=" * 65)


# %% [Cell 6: Run Pipeline]
# ==============================================================================
# CELL 6: EXECUTE TRAINING AND EVALUATION
# ==============================================================================
if __name__ == "__main__":
    # 1. Train Model (with automatic resumption if disconnected)
    latest_checkpoint = train_reid(CONFIG)

    # 2. Extract Embeddings, Save Artifacts (.npy), and Run Open-Set Evaluation
    evaluate_pipeline(CONFIG, latest_checkpoint)