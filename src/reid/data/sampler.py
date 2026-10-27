"""PK Sampler for metric learning and person re-identification."""

import random
from collections import defaultdict
from typing import Iterator, Optional, Sequence, Union

import numpy as np
from torch.utils.data import Dataset, Sampler


class PKSampler(Sampler[int]):
    """Random Identity Sampler (P x K Sampler).

    In each batch of size P * K:
    - Randomly sample P distinct identities.
    - For each identity, randomly sample K image instances.
    - If an identity has fewer than K images, sample with replacement.

    Args:
        data_source: Dataset instance with `.pids` attribute or list of identity labels.
        p: Number of distinct identities per batch (P).
        k: Number of image instances per identity (K).
        batch_size: Total batch size (P * K). If provided, p = batch_size // k.
        num_batches: Optional fixed number of batches per epoch.
                     If None, calculated from total dataset samples.
    """

    def __init__(
        self,
        data_source: Union[Dataset, Sequence[int]],
        p: Optional[int] = None,
        k: int = 4,
        batch_size: Optional[int] = None,
        num_batches: Optional[int] = None,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        if p is not None and batch_size is not None:
            if p * k != batch_size:
                raise ValueError(
                    f"Conflicting parameters: p ({p}) * k ({k}) != batch_size ({batch_size})"
                )
        elif p is None and batch_size is not None:
            if batch_size % k != 0:
                raise ValueError(
                    f"batch_size ({batch_size}) must be divisible by k ({k})"
                )
            p = batch_size // k
        elif p is not None and batch_size is None:
            batch_size = p * k
        else:
            # Default P = 16 (16 * 4 = 64 batch size)
            p = 16
            batch_size = p * k

        self.p = p
        self.k = k
        self.batch_size = batch_size

        # Extract person IDs
        if hasattr(data_source, "pids"):
            self.labels = list(data_source.pids)
        elif hasattr(data_source, "samples"):
            self.labels = [s[1] for s in data_source.samples]
        elif isinstance(data_source, (list, tuple, np.ndarray)):
            self.labels = list(data_source)
        else:
            raise TypeError(
                "data_source must be a Dataset with `.pids`/`.samples` or a sequence of labels."
            )

        self.index_pid_map = defaultdict(list)
        for idx, pid in enumerate(self.labels):
            self.index_pid_map[pid].append(idx)

        self.unique_pids = list(self.index_pid_map.keys())
        self.num_unique_pids = len(self.unique_pids)

        if self.num_unique_pids < self.p:
            raise ValueError(
                f"Number of unique identities ({self.num_unique_pids}) is less than P ({self.p})."
            )

        if num_batches is not None:
            self.num_batches = num_batches
        else:
            self.num_batches = max(1, len(self.labels) // self.batch_size)

        self.length = self.num_batches * self.batch_size

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[int]:
        batch_indices = []
        for _ in range(self.num_batches):
            selected_pids = random.sample(self.unique_pids, self.p)
            for pid in selected_pids:
                indices = self.index_pid_map[pid]
                if len(indices) >= self.k:
                    sampled = random.sample(indices, self.k)
                else:
                    sampled = np.random.choice(
                        indices, size=self.k, replace=True
                    ).tolist()
                batch_indices.extend(sampled)

        return iter(batch_indices)
