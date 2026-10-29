"""Margin-based softmax heads (CosFace and ArcFace) for deep metric learning."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginSoftmaxHead(nn.Module):
    """Normalized margin classification head supporting CosFace and ArcFace.

    Normalizes both feature vectors and class weight vectors, computes cosine similarity,
    applies additive cosine margin (CosFace) or additive angular margin (ArcFace) to
    the ground-truth target class, and scales the resulting logits by s.

    Args:
        embedding_dim: Dimensionality of input features (D).
        num_classes: Number of target classes (C).
        s: Feature scale parameter (default: 30.0).
        m: Additive margin penalty (default: 0.35 for CosFace, 0.5 for ArcFace).
        kind: Margin penalty type: 'cosface' or 'arcface'.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        s: float = 30.0,
        m: float = 0.35,
        kind: str = "cosface",
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = float(s)
        self.m = float(m)
        self.kind = kind.lower()
        if self.kind not in ("cosface", "arcface"):
            raise ValueError(
                f"Unknown margin head kind: '{kind}'. Expected 'cosface' or 'arcface'."
            )

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.set_margin(self.m)

    def set_margin(self, m: float) -> None:
        """Update effective margin m (useful for margin warmup schedules)."""
        self.m = float(m)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Input embeddings of shape (B, D).
            targets: Optional ground truth labels of shape (B,). If None or m == 0,
                     returns plain scaled cosine logits: s * cos(theta).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        norm_x = F.normalize(features, p=2, dim=1)
        norm_w = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity: cos(theta) in [-1, 1]
        cosine = F.linear(norm_x, norm_w)  # (B, C)
        self.last_unmargined_logits = self.s * cosine

        if targets is None or self.m == 0.0:
            return self.s * cosine

        eps = 1e-7
        cosine_clamped = cosine.clamp(-1.0 + eps, 1.0 - eps)

        if self.kind == "cosface":
            # CosFace: s * (cos(theta) - m) on target class
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, targets.view(-1, 1), 1.0)
            output = self.s * (cosine - one_hot * self.m)
            return output

        elif self.kind == "arcface":
            # ArcFace: cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
            sine = torch.sqrt((1.0 - torch.square(cosine_clamped)).clamp(min=eps))
            phi = cosine_clamped * self.cos_m - sine * self.sin_m
            # Thresholding when theta + m > pi for monotonicity
            phi = torch.where(cosine_clamped > self.th, phi, cosine_clamped - self.mm)

            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, targets.view(-1, 1), 1.0)
            output = self.s * torch.where(one_hot.bool(), phi, cosine)
            return output

        return self.s * cosine
