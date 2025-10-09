# src/core/models/resnet_stubs.py
"""
ResNet stubs adapted for formal verification.

Key change:
- Replace the first MaxPool2d(3,2,1) with AvgPool2d(2,2) to avoid overlapping pooling
  (auto-LiRPA requires stride == kernel_size for pooling layers).
"""

import os
import torch
import torch.nn as nn
from torchvision import models

DEFAULT_DEVICE = os.environ.get("VERIPHI_DEVICE", "cpu")


def _make_verification_friendly(model: nn.Module) -> nn.Module:
    """
    Replace the initial maxpool with a non-overlapping average pool.
    TorchVision ResNets only pool once (right after conv1/bn1/relu).
    """
    # model.maxpool is usually MaxPool2d(kernel_size=3, stride=2, padding=1)
    # Swap to AvgPool2d(2,2) so stride == kernel_size, no overlap.
    model.maxpool = nn.AvgPool2d(kernel_size=2, stride=2)
    return model


def create_resnet18(
    pretrained: bool = False,
    num_classes: int = 10,
    device: str | None = None,
) -> nn.Module:
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = _make_verification_friendly(model)
    device = device or DEFAULT_DEVICE
    return model.to(torch.device(device))


def create_resnet50(
    pretrained: bool = False,
    num_classes: int = 1000,
    device: str | None = None,
) -> nn.Module:
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = _make_verification_friendly(model)
    device = device or DEFAULT_DEVICE
    return model.to(torch.device(device))


RESNET_MODELS = {
    "resnet18": create_resnet18,
    "resnet50": create_resnet50,
}
