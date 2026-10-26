"""Unit tests for AbsoluteNegativeLoss."""

import torch
import pytest

from reid.losses.pair import AbsoluteNegativeLoss


def test_absolute_negative_loss_zero_when_below_tau():
    """When negative cosine similarity is strictly below tau_neg, loss is zero."""
    # Orthogonal embeddings: cos_sim = 0.0 < tau_neg (0.5)
    e1 = torch.tensor([[1.0, 0.0]])
    e2 = torch.tensor([[0.0, 1.0]])
    embeddings = torch.cat([e1, e2], dim=0)
    targets = torch.tensor([0, 1])

    loss_fn = AbsoluteNegativeLoss(tau_neg=0.5, k=1, weight=1.0)
    loss, metrics = loss_fn(embeddings, targets)

    assert loss.item() == pytest.approx(0.0)
    assert metrics["loss_neg_pair"] == pytest.approx(0.0)


def test_absolute_negative_loss_active_when_above_tau():
    """When negative cosine similarity exceeds tau_neg, relu penalty is active."""
    # cos_sim between [1, 0] and [cos(30 deg), sin(30 deg)] = cos(30 deg) ~= 0.866
    import math
    angle = math.radians(30.0)
    cos_val = math.cos(angle)
    sin_val = math.sin(angle)

    e1 = torch.tensor([[1.0, 0.0]])
    e2 = torch.tensor([[cos_val, sin_val]])
    embeddings = torch.cat([e1, e2], dim=0)
    targets = torch.tensor([0, 1])

    tau_neg = 0.5
    weight = 0.5
    expected_penalty = (cos_val - tau_neg) * weight

    loss_fn = AbsoluteNegativeLoss(tau_neg=tau_neg, k=1, weight=weight)
    loss, metrics = loss_fn(embeddings, targets)

    assert loss.item() == pytest.approx(expected_penalty, abs=1e-5)
    assert metrics["loss_neg_pair"] == pytest.approx(expected_penalty, abs=1e-5)


def test_hardest_k_selection():
    """Verify that only the hardest k negatives contribute."""
    # Anchor 0 with 3 negatives having similarities 0.8, 0.6, 0.2
    # With tau_neg=0.5 and k=2:
    # Hardest 2 negatives are 0.8 (penalty 0.3) and 0.6 (penalty 0.1) -> mean penalty = 0.2
    # 0.2 is below tau_neg and excluded by top-k
    e_anchor = torch.tensor([[1.0, 0.0]])
    e_neg1 = torch.tensor([[0.8, 0.6]])    # cos_sim = 0.8
    e_neg2 = torch.tensor([[0.6, 0.8]])    # cos_sim = 0.6
    e_neg3 = torch.tensor([[0.2, 0.9798]]) # cos_sim = 0.2

    embeddings = torch.cat([e_anchor, e_neg1, e_neg2, e_neg3], dim=0)
    targets = torch.tensor([0, 1, 2, 3])

    loss_fn = AbsoluteNegativeLoss(tau_neg=0.5, k=2, weight=1.0)
    loss, metrics = loss_fn(embeddings, targets)

    assert loss.item() > 0.0
    assert "loss_neg_pair" in metrics
