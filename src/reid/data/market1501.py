"""Market-1501 dataset parser and PyTorch Dataset implementation."""

import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
from torch.utils.data import Dataset

from reid.data.transforms import build_transforms


class Market1501(Dataset):
    """Market-1501 Person Re-Identification Dataset.

    Market-1501 filename format:
        [pid]_c[camid]s[seq]_[frame]_[idx].jpg
        e.g., '0002_c1s1_000451_03.jpg'
        pid: Person identity (-1 for junk, 0 for distractor, >=1 for targets)
        camid: Camera index (c1 to c6)

    Splits:
        - 'train': images from bounding_box_train/ (filtered for pid >= 0, pid != -1)
        - 'query': images from query/
        - 'gallery': images from bounding_box_test/
    """

    FILENAME_PATTERN = re.compile(r"([-\d]+)_c(\d+)")

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        relabel: Optional[bool] = None,
        image_size: Tuple[int, int] = (256, 128),
        smoke_test: bool = False,
        smoke_identities: int = 50,
        val_identities: int = 0,
        val_seed: int = 42,
    ) -> None:
        """Initialize Market-1501 dataset.

        Args:
            root_dir: Path to Market-1501 root directory (or parent directory containing it).
            split: One of 'train', 'query', 'gallery' (or 'test'), 'val_query', 'val_gallery'.
            transform: Custom image transform pipeline. If None, default transforms are built.
            relabel: If True, relabels person IDs to contiguous [0, num_classes-1].
                     Defaults to True for 'train', False for 'query'/'gallery'/'val_*'.
            image_size: Image target size (height, width) when building default transforms.
            smoke_test: Fast verification mode with limited identities.
            smoke_identities: Number of identities in smoke test mode.
            val_identities: Number of train identities held out for validation (default: 0).
            val_seed: Random seed for holding out validation identities and image assignment.
        """
        super().__init__()
        if root_dir is None:
            root_dir = os.environ.get("REID_DATA_ROOT", "data")
        self.root_dir = self._resolve_root_dir(Path(root_dir))
        self.split = "gallery" if split == "test" else split
        valid_splits = {"train", "query", "gallery", "val_query", "val_gallery"}
        if self.split not in valid_splits:
            raise ValueError(
                f"Invalid split '{split}'. Expected one of {sorted(valid_splits)}."
            )

        self.relabel = (self.split == "train") if relabel is None else relabel

        if transform is not None:
            self.transform = transform
        else:
            self.transform = build_transforms(
                image_size=image_size, is_train=(self.split == "train")
            )

        self.smoke_test = smoke_test
        self.smoke_identities = smoke_identities
        self.val_identities = val_identities
        self.val_seed = val_seed
        self.heldout_val_pids: List[int] = []
        self.train_pids: List[int] = []

        self.samples, self.num_classes, self.num_cameras = self._load_samples()

    def _resolve_root_dir(self, path: Path) -> Path:
        """Resolve dataset folder if nested or located in a parent data/ folder."""
        if (path / "bounding_box_train").is_dir():
            return path

        candidates = [
            path / "Market-1501-v15.09.15",
            path / "data" / "Market-1501-v15.09.15",
            path / "Market-1501",
            path / "market1501",
        ]
        for candidate in candidates:
            if (candidate / "bounding_box_train").is_dir():
                return candidate

        raise FileNotFoundError(
            f"Could not locate Market-1501 directory with 'bounding_box_train' in {path} or subfolders."
        )

    def _get_split_folder(self) -> str:
        if self.split in ("train", "val_query", "val_gallery"):
            return "bounding_box_train"
        if self.split == "query":
            return "query"
        return "bounding_box_test"

    def _parse_filename(self, filename: str) -> Optional[Tuple[int, int]]:
        """Extract (pid, camid) from filename."""
        match = self.FILENAME_PATTERN.match(filename)
        if not match:
            return None
        pid = int(match.group(1))
        camid = int(match.group(2))
        return pid, camid

    def _load_samples(self) -> Tuple[List[Tuple[str, int, int]], int, int]:
        folder_path = self.root_dir / self._get_split_folder()
        if not folder_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {folder_path}")

        raw_samples: List[Tuple[str, int, int]] = []
        pids_set = set()
        cams_set = set()

        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            parsed = self._parse_filename(fname)
            if parsed is None:
                continue
            pid, camid = parsed

            # For bounding_box_train (train, val_query, val_gallery), ignore junk (-1) and distractor (0)
            if self.split in ("train", "val_query", "val_gallery") and (
                pid == -1 or pid == 0
            ):
                continue

            img_path = str(folder_path / fname)
            raw_samples.append((img_path, pid, camid))
            pids_set.add(pid)
            cams_set.add(camid)

        # Handle validation hold-out when data comes from bounding_box_train
        if (
            self.split in ("train", "val_query", "val_gallery")
            and self.val_identities > 0
        ):
            import numpy as np

            all_pids = sorted(pids_set)
            if self.smoke_test:
                all_pids = all_pids[: self.smoke_identities]
                valid_smoke_pids = set(all_pids)
                raw_samples = [s for s in raw_samples if s[1] in valid_smoke_pids]
                n_val = min(self.val_identities, max(1, len(all_pids) // 5))
            else:
                n_val = min(self.val_identities, len(all_pids) - 1)
            rng = np.random.RandomState(self.val_seed)
            shuffled_pids = rng.permutation(all_pids)
            heldout_val_set = set(shuffled_pids[:n_val])
            active_train_set = set(shuffled_pids[n_val:])

            self.heldout_val_pids = [int(p) for p in sorted(heldout_val_set)]
            self.train_pids = [int(p) for p in sorted(active_train_set)]

            if self.split == "train":
                raw_samples = [s for s in raw_samples if s[1] in active_train_set]
                pids_set = active_train_set
            elif self.split in ("val_query", "val_gallery"):
                val_raw_samples = [s for s in raw_samples if s[1] in heldout_val_set]
                # Group by (pid, camid)
                from collections import defaultdict

                grouped: Dict[Tuple[int, int], List[Tuple[str, int, int]]] = (
                    defaultdict(list)
                )
                for s in val_raw_samples:
                    grouped[(s[1], s[2])].append(s)

                val_query_samples: List[Tuple[str, int, int]] = []
                val_gallery_samples: List[Tuple[str, int, int]] = []

                for (grp_pid, grp_cam), grp_imgs in sorted(grouped.items()):
                    if len(grp_imgs) >= 2:
                        grp_rng = np.random.RandomState(
                            self.val_seed + grp_pid * 1000 + grp_cam
                        )
                        q_idx = int(grp_rng.choice(len(grp_imgs)))
                        val_query_samples.append(grp_imgs[q_idx])
                        val_gallery_samples.extend(
                            [grp_imgs[j] for j in range(len(grp_imgs)) if j != q_idx]
                        )
                    else:
                        val_gallery_samples.extend(grp_imgs)

                if self.split == "val_query":
                    raw_samples = val_query_samples
                    pids_set = {s[1] for s in raw_samples}
                else:
                    raw_samples = val_gallery_samples
                    pids_set = {s[1] for s in raw_samples}
                cams_set = {s[2] for s in raw_samples}

        # Map person IDs to contiguous indices [0, num_classes-1] if relabel is requested
        if self.relabel:
            sorted_pids = sorted(pids_set)
            if self.smoke_test and self.val_identities == 0:
                sorted_pids = sorted_pids[: self.smoke_identities]
                valid_pids = set(sorted_pids)
                raw_samples = [s for s in raw_samples if s[1] in valid_pids]
            self.pid_to_label: Dict[int, int] = {
                pid: idx for idx, pid in enumerate(sorted_pids)
            }
            self.label_to_pid: Dict[int, int] = {
                idx: pid for pid, idx in self.pid_to_label.items()
            }
            final_samples = [
                (path, self.pid_to_label[pid], camid)
                for path, pid, camid in raw_samples
            ]
            num_classes = len(sorted_pids)
        else:
            self.pid_to_label = {pid: pid for pid in pids_set}
            self.label_to_pid = {pid: pid for pid in pids_set}
            final_samples = raw_samples
            num_classes = len(pids_set)

        num_cameras = len(cams_set)
        return final_samples, num_classes, num_cameras

    @property
    def pids(self) -> List[int]:
        """Return list of pids for each sample in the dataset."""
        return [sample[1] for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[object, int, int]:
        img_path, pid, camid = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, pid, camid
