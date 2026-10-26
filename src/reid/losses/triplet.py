"""Batch-hard triplet loss for metric learning and person ReID."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean_distance_matrix(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Pairwise Euclidean distance matrix with numerical stability.

    Args:
        x: Tensor of shape (B, D).
        eps: Small constant to avoid zero gradients at distance = 0.

    Returns:
        Distance matrix of shape (B, B).
    """
    # (x - y)^2 = x^2 - 2xy + y^2
    dot_prod = torch.matmul(x, x.t())
    sq_norm = torch.sum(x**2, dim=1, keepdim=True)
    dist_sq = sq_norm - 2.0 * dot_prod + sq_norm.t()
    dist_sq = torch.clamp(dist_sq, min=0.0)
    dist = torch.sqrt(dist_sq + eps)
    return dist


class BatchHardTripletLoss(nn.Module):
    """Batch-Hard Triplet Loss.

    For each anchor in the batch, selects the hardest positive (furthest same identity)
    and hardest negative (closest different identity).

    Loss = [d(a, p) - d(a, n) + margin]+

    Args:
        margin: Triplet margin (default: 0.3).
        soft_margin: If True, uses softplus log(1 + exp(d_ap - d_an)) instead of hinge.
        metric: Distance metric to use ('euclidean' or 'cosine'). Default: 'euclidean'.
    """

    def __init__(
        self,
        margin: float = 0.3,
        soft_margin: bool = False,
        metric: str = "euclidean",
    ) -> None:
        super().__init__()
        self.margin = margin
        self.soft_margin = soft_margin
        self.metric = metric

    def forward(
        self, embeddings: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute batch-hard triplet loss.

        Args:
            embeddings: Feature embeddings of shape (B, D).
            targets: Ground-truth identity labels of shape (B,).

        Returns:
            Tuple of (loss, dist_ap, dist_an).
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        if self.metric == "cosine":
            norm_emb = F.normalize(embeddings, p=2, dim=1)
            dist_mat = 1.0 - torch.matmul(norm_emb, norm_emb.t())
        else:
            dist_mat = euclidean_distance_matrix(embeddings)

        # Boolean masks
        targets_col = targets.unsqueeze(1)
        targets_row = targets.unsqueeze(0)
        mask_same = torch.eq(targets_col, targets_row)
        mask_diff = torch.ne(targets_col, targets_row)

        eye = torch.eye(batch_size, dtype=torch.bool, device=device)
        mask_pos = mask_same & (~eye)
        mask_neg = mask_diff

        # Hardest positive: max distance among positive pairs
        # If an anchor has no other positive, fallback to itself
        dist_mat_pos = dist_mat.clone()
        dist_mat_pos[~mask_pos] = -1.0
        dist_ap, _ = torch.max(dist_mat_pos, dim=1)

        # Hardest negative: min distance among negative pairs
        # Replace non-negatives with infinity
        dist_mat_neg = dist_mat.clone()
        dist_mat_neg[~mask_neg] = float("inf")
        dist_an, _ = torch.min(dist_mat_neg, dim=1)

        # Compute loss
        diff = dist_ap - dist_an
        if self.soft_margin:
            loss = F.softplus(diff).mean()
        else:
            loss = F.relu(diff + self.margin).mean()

        return loss, dist_ap, dist_an
