"""Unit tests for HardIdentityPKSampler."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from reid.data.hard_sampler import HardIdentityPKSampler


class MockReIDDataset(Dataset):
    def __init__(self, num_ids: int = 16, imgs_per_id: int = 5):
        self.samples = []
        for pid in range(num_ids):
            for i in range(imgs_per_id):
                self.samples.append((f"dummy_img_{pid}_{i}.jpg", pid, 1))

    @property
    def pids(self):
        return [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, pid, cam = self.samples[idx]
        img = torch.randn(3, 32, 16)
        return img, pid, cam


class TinyMockModel(nn.Module):
    def __init__(self, embedding_dim: int = 8):
        super().__init__()
        self.conv = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, embedding_dim)

    def forward(self, x):
        feat = self.conv(x).view(x.size(0), -1)
        feat = self.fc(feat)
        return torch.nn.functional.normalize(feat, p=2, dim=1)


def test_batch_size_and_no_duplicate_identities():
    num_ids = 20
    dataset = MockReIDDataset(num_ids=num_ids, imgs_per_id=6)
    p = 8
    k = 3
    group_size = 4

    sampler = HardIdentityPKSampler(
        data_source=dataset,
        p=p,
        k=k,
        group_size=group_size,
        mining_start_epoch=0,
    )

    # Set mock similarity matrix
    sim_mat = np.random.randn(num_ids, num_ids)
    sim_mat = (sim_mat + sim_mat.T) / 2.0
    np.fill_diagonal(sim_mat, -1e9)
    sampler.similarity_matrix = sim_mat
    sampler.set_epoch(1)

    for batch in sampler:
        assert len(batch) == p * k
        batch_pids = [dataset.samples[idx][1] for idx in batch]
        unique_batch_pids = set(batch_pids)
        assert len(unique_batch_pids) == p  # Exactly p distinct identities, no duplicates


def test_fallback_to_pk_before_mining_start_epoch():
    num_ids = 16
    dataset = MockReIDDataset(num_ids=num_ids, imgs_per_id=4)
    sampler = HardIdentityPKSampler(
        data_source=dataset,
        p=8,
        k=2,
        mining_start_epoch=10,
    )
    # At epoch 5 (< 10), even if similarity matrix is present, it uses plain PK
    sampler.similarity_matrix = np.ones((num_ids, num_ids))
    sampler.set_epoch(5)

    for batch in sampler:
        assert len(batch) == 8 * 2
        batch_pids = [dataset.samples[idx][1] for idx in batch]
        assert len(set(batch_pids)) == 8


def test_synthetic_neighborhood_clustering():
    """Verify that with synthetic clusters, sampled neighborhoods group mutual nearest neighbors."""
    # 8 identities forming 2 distinct clusters of 4:
    # Cluster A: ids 0, 1, 2, 3 (similarity = 0.99 with each other, -0.99 with cluster B)
    # Cluster B: ids 4, 5, 6, 7 (similarity = 0.99 with each other, -0.99 with cluster A)
    num_ids = 8
    dataset = MockReIDDataset(num_ids=num_ids, imgs_per_id=4)

    sim_mat = np.full((num_ids, num_ids), -0.99)
    sim_mat[:4, :4] = 0.99
    sim_mat[4:, 4:] = 0.99
    np.fill_diagonal(sim_mat, -1e9)

    sampler = HardIdentityPKSampler(
        data_source=dataset,
        p=8,
        k=2,
        group_size=4,
        m_neighbors=4,
        mining_start_epoch=0,
    )
    sampler.similarity_matrix = sim_mat
    sampler.set_epoch(0)

    for batch in sampler:
        batch_pids = [dataset.samples[idx][1] for idx in batch]
        distinct_pids = []
        for pid in batch_pids:
            if pid not in distinct_pids:
                distinct_pids.append(pid)

        # The first 4 must be one cluster, the next 4 the other cluster
        group1 = set(distinct_pids[:4])
        group2 = set(distinct_pids[4:])

        assert group1 == {0, 1, 2, 3} or group1 == {4, 5, 6, 7}
        assert group2 == {0, 1, 2, 3} or group2 == {4, 5, 6, 7}


def test_refresh_centroids_on_cpu():
    """Verify centroid extraction works on CPU with a tiny model."""
    dataset = MockReIDDataset(num_ids=8, imgs_per_id=4)
    model = TinyMockModel(embedding_dim=16)

    sampler = HardIdentityPKSampler(data_source=dataset, p=4, k=2, group_size=2)
    assert sampler.similarity_matrix is None

    sampler.refresh_centroids(model, device=torch.device("cpu"), max_images_per_id=2)
    assert sampler.similarity_matrix is not None
    assert sampler.similarity_matrix.shape == (8, 8)
    # Diagonal should be -1e9
    assert np.all(np.diag(sampler.similarity_matrix) <= -1e8)
