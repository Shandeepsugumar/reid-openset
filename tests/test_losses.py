"""Unit tests for baseline loss and batch-hard triplet loss."""

import torch
from reid.losses import BaselineReIDLoss, BatchHardTripletLoss


def test_batch_hard_triplet_loss() -> None:
    # 2 identities, 2 samples each -> 4 samples
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],  # Identity 0
            [0.0, 1.0],
            [0.1, 0.9],  # Identity 1
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    criterion = BatchHardTripletLoss(margin=0.3)
    loss, dist_ap, dist_an = criterion(embeddings, targets)

    assert loss.dim() == 0
    assert loss.item() >= 0.0
    assert dist_ap.shape == (4,)
    assert dist_an.shape == (4,)
    # For identity 0, positive dist should be much smaller than negative dist
    assert dist_ap.mean() < dist_an.mean()


def test_baseline_combined_loss() -> None:
    batch_size = 8
    num_classes = 10
    embedding_dim = 512

    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    embeddings = torch.randn(batch_size, embedding_dim, requires_grad=True)
    # 4 identities, 2 samples each
    targets = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)

    criterion = BaselineReIDLoss(margin=0.3, label_smoothing=0.1)
    loss, metrics = criterion(logits, embeddings, targets)

    assert loss.dim() == 0
    assert "loss" in metrics
    assert "loss_ce" in metrics
    assert "loss_triplet" in metrics

    loss.backward()
    assert logits.grad is not None
    assert embeddings.grad is not None
