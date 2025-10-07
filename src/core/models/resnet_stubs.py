# src/core/models/resnet_stubs.py
"""
ResNet Stubs for Verification
"""

import os
import torch
import torch.nn as nn
from torchvision import models

DEFAULT_DEVICE = os.environ.get("VERIPHI_DEVICE", "cpu")


def create_resnet18(pretrained: bool = False, num_classes: int = 10, device: str | None = None) -> nn.Module:
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    device = device or DEFAULT_DEVICE
    return model.to(torch.device(device))


def create_resnet50(pretrained: bool = False, num_classes: int = 1000, device: str | None = None) -> nn.Module:
    model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    device = device or DEFAULT_DEVICE
    return model.to(torch.device(device))


RESNET_MODELS = {"resnet18": create_resnet18, "resnet50": create_resnet50}
