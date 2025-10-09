# src/core/models/__init__.py
"""
Neural Network Models Module
"""

import os
import torch

from .test_models import (
    SimpleLinearNet,
    SimpleConvNet,
    TinyNet,
    DeepLinearNet,
    initialize_model_weights,
    load_pretrained_model,
    create_sample_input,
    create_model_from_config,
    MODEL_CONFIGS,
)
from .resnet_stubs import create_resnet18, create_resnet50
from .trm_adapter import create_trm_mlp

# default device from environment
DEFAULT_DEVICE = os.environ.get("VERIPHI_DEVICE", "cpu")

# ---------------------------------------------------------------------
# Unified factory
# ---------------------------------------------------------------------
def create_test_model(model_type: str = "tiny", device: str | None = None, **kwargs) -> torch.nn.Module:
    """
    Create a test or reference model by name.

    model_type: one of
        ["tiny", "linear", "conv", "deep", "resnet18", "resnet50", "trm-mlp"]
    """
    device = device or DEFAULT_DEVICE
    model_type = model_type.lower()

    if model_type == "tiny":
        from .test_models import TinyNet
        model = TinyNet(**kwargs)
    elif model_type == "linear":
        from .test_models import SimpleLinearNet
        model = SimpleLinearNet(**kwargs)
    elif model_type == "conv":
        from .test_models import SimpleConvNet
        model = SimpleConvNet(**kwargs)
    elif model_type == "deep":
        from .test_models import DeepLinearNet
        model = DeepLinearNet(**kwargs)
    elif model_type == "resnet18":
        model = create_resnet18(pretrained=False, num_classes=10)
    elif model_type == "resnet50":
        model = create_resnet50(pretrained=False, num_classes=1000)
    elif model_type == "trm-mlp":
        # ~7M parameters; tune dims to GPU capacity
        model = create_trm_mlp(
            x_dim=512, y_dim=512, z_dim=512,
            hidden=1024, num_classes=10,
            H_cycles=3, L_cycles=4
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(torch.device(device))


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
__all__ = [
    # Model classes
    "SimpleLinearNet",
    "SimpleConvNet",
    "TinyNet",
    "DeepLinearNet",
    # Factory functions
    "create_test_model",
    "create_model_from_config",
    "create_sample_input",
    # Utilities
    "initialize_model_weights",
    "load_pretrained_model",
    # Configurations
    "MODEL_CONFIGS",
    # ResNets
    "create_resnet18",
    "create_resnet50",
    # TRM adapter
    "create_trm_mlp",
    # Device default
    "DEFAULT_DEVICE",
]
