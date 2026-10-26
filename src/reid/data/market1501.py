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
    ) -> None:
        """Initialize Market-1501 dataset.

        Args:
            root_dir: Path to Market-1501 root directory (or parent directory containing it).
            split: One of 'train', 'query', 'gallery' (or 'test').
            transform: Custom image transform pipeline. If None, default transforms are built.
            relabel: If True, relabels person IDs to contiguous [0, num_classes-1].
                     Defaults to True for 'train', False for 'query'/'gallery'.
            image_size: Image target size (height, width) when building default transforms.
        """
        super().__init__()
        if root_dir is None:
            root_dir = os.environ.get("REID_DATA_ROOT", "data")
        self.root_dir = self._resolve_root_dir(Path(root_dir))
        self.split = "gallery" if split == "test" else split
        if self.split not in {"train", "query", "gallery"}:
            raise ValueError(
                f"Invalid split '{split}'. Expected 'train', 'query', or 'gallery' (or 'test')."
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
        if self.split == "train":
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

            # For training, ignore junk (-1) and distractor (0) identities
            if self.split == "train" and (pid == -1 or pid == 0):
                continue

            img_path = str(folder_path / fname)
            raw_samples.append((img_path, pid, camid))
            pids_set.add(pid)
            cams_set.add(camid)

        # Map person IDs to contiguous indices [0, num_classes-1] if relabel is requested
        if self.relabel:
            sorted_pids = sorted(pids_set)
            if self.smoke_test:
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
