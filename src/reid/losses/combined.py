from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from reid.losses.pair import AbsoluteNegativeLoss
from reid.losses.triplet import BatchHardTripletLoss


class BaselineReIDLoss(nn.Module):
    """Combined Cross-Entropy ID loss, Batch-Hard Triplet loss, and optional Absolute Negative loss.

    Args:
        margin: Margin for triplet loss (default: 0.3).
        label_smoothing: Smoothing value for cross entropy loss (default: 0.1).
        weight_ce: Loss weight for cross-entropy ID loss (default: 1.0).
        weight_triplet: Loss weight for triplet loss (default: 1.0).
        soft_margin: Whether to use softplus in triplet loss.
        neg_pair_loss: Optional AbsoluteNegativeLoss module.
    """

    def __init__(
        self,
        margin: float = 0.3,
        label_smoothing: float = 0.0,
        weight_ce: float = 1.0,
        weight_triplet: float = 1.0,
        soft_margin: bool = False,
        neg_pair_loss: Optional[AbsoluteNegativeLoss] = None,
    ) -> None:
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_triplet = weight_triplet
        self.ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.triplet_loss_fn = BatchHardTripletLoss(
            margin=margin, soft_margin=soft_margin
        )
        self.neg_pair_loss = neg_pair_loss

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.

        Args:
            logits: Classifier logits of shape (B, num_classes).
            embeddings: Feature embeddings of shape (B, D).
            targets: Identity ground truth of shape (B,).

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        ce_loss = self.ce_loss_fn(logits, targets)
        triplet_loss, dist_ap, dist_an = self.triplet_loss_fn(embeddings, targets)

        total_loss = self.weight_ce * ce_loss + self.weight_triplet * triplet_loss

        metrics: Dict[str, float] = {
            "loss": float(total_loss.detach().item()),
            "loss_ce": float(ce_loss.detach().item()),
            "loss_triplet": float(triplet_loss.detach().item()),
            "dist_ap": float(dist_ap.mean().detach().item()),
            "dist_an": float(dist_an.mean().detach().item()),
        }

        if self.neg_pair_loss is not None:
            loss_pair, pair_metrics = self.neg_pair_loss(embeddings, targets)
            total_loss = total_loss + loss_pair
            metrics["loss"] = float(total_loss.detach().item())
            metrics.update(pair_metrics)

        return total_loss, metrics
