"""Unit tests for ReIDBackbone and embedding outputs."""

import pytest
import torch

from reid.models.backbone import ReIDBackbone


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_output_embedding_shape_and_normalization(batch_size: int) -> None:
    """Verify that backbone produces (batch_size, 512) L2-normalized embeddings."""
    model = ReIDBackbone(
        backbone_name="resnet18",
        embedding_dim=512,
        num_classes=751,
        pretrained=False,
    )
    model.eval()

    inputs = torch.randn(batch_size, 3, 256, 128)
    with torch.no_grad():
        embeddings = model(inputs)

    # Verify shape
    assert embeddings.shape == (
        batch_size,
        512,
    ), f"Expected shape ({batch_size}, 512), got {embeddings.shape}"

    # Verify L2 normalization: ||e_i||_2 == 1.0
    norms = torch.norm(embeddings, p=2, dim=1)
    expected_norms = torch.ones(batch_size)
    assert torch.allclose(
        norms, expected_norms, atol=1e-5
    ), f"Embeddings are not L2 normalized: {norms}"


def test_training_mode_outputs() -> None:
    """Verify that training mode returns logits and normalized features."""
    num_classes = 100
    batch_size = 6
    model = ReIDBackbone(
        backbone_name="resnet18",
        embedding_dim=512,
        num_classes=num_classes,
        pretrained=False,
    )
    model.train()

    inputs = torch.randn(batch_size, 3, 256, 128)
    logits, feat = model(inputs)

    assert logits.shape == (batch_size, num_classes)
    assert feat.shape == (batch_size, 512)

    # feat should also be L2 normalized
    norms = torch.norm(feat, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)


def test_configurable_embedding_dim() -> None:
    """Verify custom embedding dimension support (e.g. 256)."""
    batch_size = 2
    model = ReIDBackbone(
        backbone_name="resnet18",
        embedding_dim=256,
        pretrained=False,
    )
    model.eval()

    inputs = torch.randn(batch_size, 3, 256, 128)
    with torch.no_grad():
        out = model(inputs)

    assert out.shape == (batch_size, 256)
    norms = torch.norm(out, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)
