"""Compact backbone wrappers with BatchNorm neck and L2 normalization."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from reid.losses.margin import MarginSoftmaxHead


def weights_init_kaiming(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class ReIDBackbone(nn.Module):
    """Backbone wrapper for Person Re-Identification.

    Features:
    - Configurable torchvision CNN backbone (default: resnet18).
    - Global Adaptive Average Pooling.
    - 512-dim embedding head.
    - BatchNorm Neck (BNNeck) for joint classification and metric learning.
    - L2-normalized output for distance metric computation.

    Args:
        backbone_name: Name of torchvision backbone (e.g., 'resnet18', 'resnet34', 'resnet50').
        embedding_dim: Dimensionality of output embedding (default: 512).
        num_classes: Number of training identities for the classification head.
                     If None or <=0, classification head is omitted.
        pretrained: Whether to load ImageNet pre-trained weights.
        neck_feat: Which feature to normalize for inference ('before' or 'after' BNNeck).
                   Default is 'after'.
        classifier_type: Classification head type ('linear', 'cosface', 'arcface'). Default: 'linear'.
        margin_s: Feature scale for margin softmax (default: 30.0).
        margin_m: Margin penalty for margin softmax (default: 0.35).
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        embedding_dim: int = 512,
        num_classes: Optional[int] = None,
        pretrained: bool = False,
        neck_feat: str = "after",
        classifier_type: str = "linear",
        margin_s: float = 30.0,
        margin_m: float = 0.35,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.neck_feat = neck_feat
        self.classifier_type = classifier_type.lower()
        self.margin_s = float(margin_s)
        self.margin_m = float(margin_m)

        # Load torchvision model
        if not hasattr(tv_models, backbone_name):
            raise ValueError(f"Unsupported torchvision backbone: '{backbone_name}'")

        model_fn = getattr(tv_models, backbone_name)
        try:
            # Modern torchvision syntax
            weights = "DEFAULT" if pretrained else None
            base_model = model_fn(weights=weights)
        except (TypeError, ValueError):
            # Fallback for older torchvision
            base_model = model_fn(pretrained=pretrained)

        # Truncate classifier/fc and extract convolutional stages
        if hasattr(base_model, "fc"):
            in_features = base_model.fc.in_features
            # Remove avgpool and fc from base model
            self.conv_base = nn.Sequential(*list(base_model.children())[:-2])
        else:
            raise NotImplementedError(
                f"Backbone '{backbone_name}' does not have an 'fc' layer."
            )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Embedding projection head (projects backbone dim -> embedding_dim)
        if in_features != embedding_dim:
            self.head = nn.Linear(in_features, embedding_dim, bias=False)
            self.head.apply(weights_init_kaiming)
        else:
            self.head = nn.Identity()

        # BatchNorm neck (BNNeck)
        self.bnneck = nn.BatchNorm1d(embedding_dim)
        self.bnneck.bias.requires_grad_(False)
        self.bnneck.apply(weights_init_kaiming)

        # Classification head for ID loss
        if num_classes is not None and num_classes > 0:
            if self.classifier_type in ("cosface", "arcface"):
                self.classifier = MarginSoftmaxHead(
                    embedding_dim=embedding_dim,
                    num_classes=num_classes,
                    s=self.margin_s,
                    m=self.margin_m,
                    kind=self.classifier_type,
                )
            else:
                self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
                self.classifier.apply(weights_init_classifier)
        else:
            self.classifier = None

    def extract_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract raw embedding, BNNeck embedding, and L2-normalized embedding.

        Returns:
            Tuple of (feat, feat_bn, norm_feat).
        """
        x = self.conv_base(x)
        x = self.gap(x)
        feat = torch.flatten(x, 1)
        feat = self.head(feat)
        feat_bn = self.bnneck(feat)

        target_feat = feat_bn if self.neck_feat == "after" else feat
        norm_feat = F.normalize(target_feat, p=2, dim=1)

        return feat, feat_bn, norm_feat

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_all: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], dict]:
        """Forward pass.

        - When return_all=True: returns dict {"feat": feat, "feat_bn": feat_bn, "norm_feat": norm_feat, "logits": logits}.
        - In training mode (self.training=True and classifier is present): returns (logits, norm_feat).
        - In evaluation mode: returns norm_feat (L2-normalized 512-dim embedding).
        """
        feat, feat_bn, norm_feat = self.extract_features(x)

        if self.classifier is not None:
            if isinstance(self.classifier, MarginSoftmaxHead):
                logits = self.classifier(feat_bn, targets)
            else:
                logits = self.classifier(feat_bn)
        else:
            logits = None

        if return_all:
            return {
                "feat": feat,
                "feat_bn": feat_bn,
                "norm_feat": norm_feat,
                "logits": logits,
            }

        if self.training and logits is not None:
            return logits, norm_feat

        return norm_feat
