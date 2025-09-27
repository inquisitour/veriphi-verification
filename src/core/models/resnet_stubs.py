# src/core/models/resnet_stubs.py
"""
ResNet Stubs for Verification

Provides lightweight ResNet-18 and ResNet-50 stubs compatible with our verification system.
These use torchvision.models as a backend but are wrapped for easy integration.
"""

import torch
import torch.nn as nn
from torchvision import models


def create_resnet18(pretrained: bool = False, num_classes: int = 10) -> nn.Module:
    """
    Create a ResNet-18 model for CIFAR-10 or similar datasets.

    Args:
        pretrained: If True, loads pretrained ImageNet weights
        num_classes: Number of output classes

    Returns:
        nn.Module: ResNet-18 model
    """
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    # Replace the final layer to match our number of classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def create_resnet50(pretrained: bool = False, num_classes: int = 1000) -> nn.Module:
    """
    Create a ResNet-50 model for ImageNet or similar datasets.

    Args:
        pretrained: If True, loads pretrained ImageNet weights
        num_classes: Number of output classes

    Returns:
        nn.Module: ResNet-50 model
    """
    model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
    # Replace the final layer to match our number of classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# Convenience map (optional, for consistency with create_test_model)
RESNET_MODELS = {
    "resnet18": create_resnet18,
    "resnet50": create_resnet50,
}
