# src/core/models/test_models.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

DEFAULT_DEVICE = os.environ.get("VERIPHI_DEVICE", "cpu")


# ---------------------------------------------------------------------
# model definitions
# ---------------------------------------------------------------------
class SimpleLinearNet(nn.Module):
    def __init__(self, input_size: int = 784, hidden_size: int = 100, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class SimpleConvNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class TinyNet(nn.Module):
    def __init__(self, input_size: int = 10, num_classes: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class DeepLinearNet(nn.Module):
    def __init__(self, input_size: int = 784, hidden_sizes: list = None, num_classes: int = 10):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128, 64]

        layers = [nn.Flatten()]
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------------------
# factory & helpers
# ---------------------------------------------------------------------
def create_test_model(model_type: str = "linear", device: Optional[str] = None, **kwargs) -> nn.Module:
    model_map = {
        "linear": SimpleLinearNet,
        "conv": SimpleConvNet,
        "tiny": TinyNet,
        "deep": DeepLinearNet,
    }
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_map.keys())}")
    model_class = model_map[model_type]
    model = model_class(**kwargs)
    device = device or DEFAULT_DEVICE
    return model.to(torch.device(device))


def initialize_model_weights(model: nn.Module, method: str = "xavier") -> nn.Module:
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if method == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif method == "kaiming":
                nn.init.kaiming_uniform_(m.weight)
            elif method == "normal":
                nn.init.normal_(m.weight, 0, 0.1)
            elif method == "zero":
                nn.init.zeros_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)
    return model


def load_pretrained_model(model_path: str) -> nn.Module:
    if model_path.endswith((".pt", ".pth")):
        model = torch.load(model_path, map_location="cpu")
        model.eval()
        return model
    elif model_path.endswith(".onnx"):
        raise NotImplementedError("ONNX loading not implemented yet")
    else:
        raise ValueError(f"Unsupported model format: {model_path}")


def create_sample_input(model_type: str, batch_size: int = 1, device: Optional[str] = None) -> torch.Tensor:
    device = device or DEFAULT_DEVICE
    if model_type in ["linear", "deep"]:
        x = torch.randn(batch_size, 1, 28, 28)
    elif model_type == "conv":
        x = torch.randn(batch_size, 1, 28, 28)
    elif model_type == "tiny":
        x = torch.randn(batch_size, 10)
    else:
        x = torch.randn(batch_size, 1, 28, 28)
    return x.to(torch.device(device))


MODEL_CONFIGS = {
    "mnist_simple": {"type": "linear", "input_size": 784, "hidden_size": 100, "num_classes": 10},
    "mnist_conv": {"type": "conv", "num_classes": 10},
    "cifar_simple": {"type": "linear", "input_size": 3072, "hidden_size": 256, "num_classes": 10},
    "toy_problem": {"type": "tiny", "input_size": 10, "num_classes": 3},
    "deep_network": {"type": "deep", "input_size": 784, "hidden_sizes": [512, 256, 128, 64], "num_classes": 10},
}


def create_model_from_config(config_name: str, device: Optional[str] = None) -> nn.Module:
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(MODEL_CONFIGS.keys())}")
    cfg = MODEL_CONFIGS[config_name].copy()
    mtype = cfg.pop("type")
    return create_test_model(mtype, device=device or DEFAULT_DEVICE, **cfg)
