"""Unit tests for MarginSoftmaxHead (CosFace and ArcFace)."""

import torch
import pytest

from reid.losses.margin import MarginSoftmaxHead


def test_output_shape():
    batch_size = 8
    dim = 64
    num_classes = 20

    for kind in ("cosface", "arcface"):
        head = MarginSoftmaxHead(embedding_dim=dim, num_classes=num_classes, s=30.0, m=0.35, kind=kind)
        features = torch.randn(batch_size, dim)
        targets = torch.randint(0, num_classes, (batch_size,))

        # With targets
        logits = head(features, targets)
        assert logits.shape == (batch_size, num_classes)

        # Without targets (eval mode)
        logits_eval = head(features)
        assert logits_eval.shape == (batch_size, num_classes)


def test_gradient_flows():
    batch_size = 4
    dim = 32
    num_classes = 10

    for kind in ("cosface", "arcface"):
        head = MarginSoftmaxHead(embedding_dim=dim, num_classes=num_classes, s=30.0, m=0.35, kind=kind)
        features = torch.randn(batch_size, dim, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size,))

        logits = head(features, targets)
        loss = logits.sum()
        loss.backward()

        assert features.grad is not None
        assert head.weight.grad is not None
        assert torch.all(torch.isfinite(features.grad))
        assert torch.all(torch.isfinite(head.weight.grad))


def test_target_logit_lower_when_m_greater_than_zero():
    batch_size = 4
    dim = 32
    num_classes = 10
    s = 30.0

    for kind in ("cosface", "arcface"):
        head = MarginSoftmaxHead(embedding_dim=dim, num_classes=num_classes, s=s, m=0.35, kind=kind)
        features = torch.randn(batch_size, dim)
        targets = torch.tensor([0, 1, 2, 3])

        # Plain scaled cosine (without targets / m=0)
        plain_logits = head(features, targets=None)

        # Logits with margin applied to target class
        margin_logits = head(features, targets=targets)

        # For target classes, margin_logits must be strictly less than plain_logits
        for i in range(batch_size):
            target_class = targets[i].item()
            assert margin_logits[i, target_class] < plain_logits[i, target_class]

            # For non-target classes, logits must be identical
            for c in range(num_classes):
                if c != target_class:
                    assert margin_logits[i, c].item() == pytest.approx(plain_logits[i, c].item(), abs=1e-5)


def test_m_zero_equals_plain_cosine_softmax():
    batch_size = 4
    dim = 32
    num_classes = 10
    s = 30.0

    for kind in ("cosface", "arcface"):
        head = MarginSoftmaxHead(embedding_dim=dim, num_classes=num_classes, s=s, m=0.0, kind=kind)
        features = torch.randn(batch_size, dim)
        targets = torch.tensor([0, 1, 2, 3])

        logits_with_targets = head(features, targets=targets)
        logits_without_targets = head(features, targets=None)

        assert torch.allclose(logits_with_targets, logits_without_targets, atol=1e-6)
