"""Identity-level hard-negative batch sampler for Person Re-Identification."""

import os
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


class HardIdentityPKSampler(Sampler[List[int]]):
    """Batch sampler that forms batches from clusters of visually similar identities.

    Instead of purely uniform random identity sampling, batches are constructed from
    P/group_size neighborhoods of mutually look-alike identities mined via model embeddings.

    Args:
        data_source: Dataset providing samples (and pids property or indexing).
        p: Number of distinct identities per batch (default: 16).
        k: Number of instances per identity (default: 4).
        group_size: Number of look-alike identities per neighborhood (default: 4).
        m_neighbors: Candidate pool size for nearest neighbors (default: 3 * group_size).
        refresh_every: Epoch interval to re-extract centroids (default: 5).
        mining_start_epoch: Epoch at which hard identity mining starts (default: 10).
                            Prior epochs use plain random PK sampling.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Dataset,
        p: int = 16,
        k: int = 4,
        group_size: int = 4,
        m_neighbors: Optional[int] = None,
        refresh_every: int = 5,
        mining_start_epoch: int = 10,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_source = data_source
        self.p = int(p)
        self.k = int(k)
        self.group_size = int(group_size)
        self.m_neighbors = (
            int(m_neighbors) if m_neighbors is not None else 3 * self.group_size
        )
        self.refresh_every = int(refresh_every)
        self.mining_start_epoch = int(mining_start_epoch)
        self.seed = int(seed)

        if self.p % self.group_size != 0:
            raise ValueError(
                f"p ({self.p}) must be divisible by group_size ({self.group_size})"
            )

        self.num_groups = self.p // self.group_size
        self.current_epoch = 0
        self.similarity_matrix: Optional[np.ndarray] = None

        self.pid_to_indices: Dict[int, List[int]] = defaultdict(list)
        if hasattr(data_source, "pids"):
            all_pids = data_source.pids
            for idx, pid in enumerate(all_pids):
                self.pid_to_indices[pid].append(idx)
        elif hasattr(data_source, "samples"):
            for idx, sample in enumerate(data_source.samples):
                pid = sample[1]
                self.pid_to_indices[pid].append(idx)
        else:
            for idx in range(len(data_source)):
                _, pid, _ = data_source[idx]
                self.pid_to_indices[pid].append(idx)

        self.identities = sorted(self.pid_to_indices.keys())
        self.num_identities = len(self.identities)
        self.pid_to_id_idx: Dict[int, int] = {
            pid: i for i, pid in enumerate(self.identities)
        }

    def set_epoch(self, epoch: int) -> None:
        """Update current training epoch."""
        self.current_epoch = int(epoch)

    def refresh_centroids(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        max_images_per_id: int = 10,
        image_size: Sequence[int] = (256, 128),
    ) -> None:
        """Embed up to 10 images per train identity with model in eval mode without augmentation."""
        was_training = model.training
        model.eval()

        if device is None:
            device = next(model.parameters()).device

        eval_transform = transforms.Compose(
            [
                transforms.Resize(tuple(image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        centroids_list: List[torch.Tensor] = []

        with torch.no_grad():
            for pid in self.identities:
                indices = self.pid_to_indices[pid]
                selected_indices = indices[:max_images_per_id]

                batch_tensors = []
                for idx in selected_indices:
                    img_tensor = None
                    if hasattr(self.data_source, "samples"):
                        raw_item = self.data_source.samples[idx]
                        if isinstance(raw_item[0], str) and os.path.isfile(raw_item[0]):
                            img = Image.open(raw_item[0]).convert("RGB")
                            img_tensor = eval_transform(img)

                    if img_tensor is None:
                        item = self.data_source[idx]
                        img_data = item[0]
                        if isinstance(img_data, torch.Tensor):
                            img_tensor = img_data
                        elif isinstance(img_data, Image.Image):
                            img_tensor = eval_transform(img_data)
                        elif isinstance(img_data, str) and os.path.isfile(img_data):
                            img_tensor = eval_transform(
                                Image.open(img_data).convert("RGB")
                            )

                    if img_tensor is not None:
                        batch_tensors.append(img_tensor)

                if len(batch_tensors) == 0:
                    continue

                img_batch = torch.stack(batch_tensors, dim=0).to(device)
                feats = model(img_batch)
                centroid = feats.mean(dim=0, keepdim=True)
                centroid = F.normalize(centroid, p=2, dim=1)
                centroids_list.append(centroid.cpu())

        all_centroids = torch.cat(centroids_list, dim=0)
        sim_mat = torch.mm(all_centroids, all_centroids.T).numpy()

        np.fill_diagonal(sim_mat, -1e9)
        self.similarity_matrix = sim_mat

        if was_training:
            model.train()

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed + self.current_epoch)

        use_hard_mining = (
            self.current_epoch >= self.mining_start_epoch
            and self.similarity_matrix is not None
        )

        num_batches = max(1, self.num_identities // self.p)

        for _ in range(num_batches):
            if not use_hard_mining:
                shuffled_pids = rng.permutation(self.identities)
                batch_pids = list(shuffled_pids[: self.p])
            else:
                available_pids = set(self.identities)
                batch_pids = []

                for _ in range(self.num_groups):
                    if len(available_pids) == 0:
                        break

                    avail_list = sorted(available_pids)
                    anchor = avail_list[rng.choice(len(avail_list))]
                    batch_pids.append(anchor)
                    available_pids.remove(anchor)

                    needed = self.group_size - 1
                    if needed > 0 and len(available_pids) > 0:
                        anchor_idx = self.pid_to_id_idx[anchor]
                        sim_scores = self.similarity_matrix[anchor_idx]

                        ranked_indices = np.argsort(-sim_scores)
                        candidate_pool = [
                            self.identities[idx]
                            for idx in ranked_indices
                            if self.identities[idx] in available_pids
                        ][: self.m_neighbors]

                        if len(candidate_pool) < needed:
                            candidate_pool = [
                                self.identities[idx]
                                for idx in ranked_indices
                                if self.identities[idx] in available_pids
                            ]

                        sample_count = min(needed, len(candidate_pool))
                        chosen_neighbors = rng.choice(
                            candidate_pool, size=sample_count, replace=False
                        ).tolist()

                        batch_pids.extend(chosen_neighbors)
                        for nb in chosen_neighbors:
                            available_pids.remove(nb)

                if len(batch_pids) < self.p:
                    rem_candidates = [
                        pid for pid in self.identities if pid not in set(batch_pids)
                    ]
                    needed_extra = self.p - len(batch_pids)
                    if len(rem_candidates) >= needed_extra:
                        batch_pids.extend(
                            rng.choice(
                                rem_candidates, size=needed_extra, replace=False
                            ).tolist()
                        )

            batch_indices = []
            for pid in batch_pids[: self.p]:
                indices = self.pid_to_indices[pid]
                if len(indices) >= self.k:
                    sampled = rng.choice(indices, size=self.k, replace=False).tolist()
                else:
                    sampled = rng.choice(indices, size=self.k, replace=True).tolist()
                batch_indices.extend(sampled)

            yield batch_indices

    def __len__(self) -> int:
        return max(1, self.num_identities // self.p)
