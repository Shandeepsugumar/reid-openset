"""Absolute negative-pair and positive-pair loss for bounded metric space."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbsoluteNegativeLoss(nn.Module):
    """Bounds absolute similarity of negative and positive pairs.

    For each anchor in the batch:
    - Mines hardest-k negative instances (highest cosine similarity).
    - Applies penalty: relu(cos_sim_neg - tau_neg).
    - Optionally applies penalty to positives: relu(tau_pos - cos_sim_pos).

    Args:
        tau_neg: Upper threshold for negative similarity (default: 0.5).
        k: Number of hardest negatives per anchor (default: 3).
        weight: Loss weight for negative penalty (default: 0.5).
        tau_pos: Optional lower threshold for positive similarity (default: None).
        pos_weight: Loss weight for positive penalty (default: 0.0).
    """

    def __init__(
        self,
        tau_neg: float = 0.5,
        k: int = 3,
        weight: float = 0.5,
        tau_pos: Optional[float] = None,
        pos_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.tau_neg = float(tau_neg)
        self.k = int(k)
        self.weight = float(weight)
        self.tau_pos = float(tau_pos) if tau_pos is not None else None
        self.pos_weight = float(pos_weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute absolute pair loss.

        Args:
            embeddings: L2-normalized or unnormalized embeddings of shape (B, D).
            targets: Identity ground truth labels of shape (B,).

        Returns:
            Tuple of (loss_tensor, metrics_dict).
        """
        b_size = embeddings.size(0)
        device = embeddings.device

        if b_size <= 1:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, {"loss_neg_pair": 0.0}

        norm_feats = F.normalize(embeddings, p=2, dim=1)
        sim_mat = torch.matmul(norm_feats, norm_feats.T)  # (B, B)

        targets_col = targets.view(-1, 1)
        same_id = targets_col == targets_col.T
        diff_id = ~same_id

        neg_sims = sim_mat.clone()
        neg_sims.masked_fill_(same_id, -2.0)

        num_neg_per_row = diff_id.sum(dim=1)
        max_k = (
            min(self.k, int(num_neg_per_row.max().item()))
            if num_neg_per_row.max() > 0
            else 0
        )

        if max_k > 0:
            topk_neg_sims, _ = torch.topk(neg_sims, k=max_k, dim=1)
            valid_mask = topk_neg_sims > -1.5
            neg_penalties = F.relu(topk_neg_sims - self.tau_neg)
            if valid_mask.any():
                loss_neg = neg_penalties[valid_mask].mean()
            else:
                loss_neg = torch.tensor(0.0, device=device)
        else:
            loss_neg = torch.tensor(0.0, device=device)

        loss_pos = torch.tensor(0.0, device=device)
        if self.tau_pos is not None and self.pos_weight > 0.0:
            eye_mask = torch.eye(b_size, dtype=torch.bool, device=device)
            pos_mask = same_id & (~eye_mask)
            if pos_mask.any():
                pos_sims = sim_mat[pos_mask]
                pos_penalties = F.relu(self.tau_pos - pos_sims)
                loss_pos = pos_penalties.mean()

        total_loss = self.weight * loss_neg + self.pos_weight * loss_pos

        metrics = {
            "loss_neg_pair": float(total_loss.detach().item()),
        }
        return total_loss, metrics
