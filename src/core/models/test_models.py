# src/core/models/test_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SimpleLinearNet(nn.Module):
    """Simple fully connected network for testing"""
    
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
    """Simple CNN for testing"""
    
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
    """Very small network for quick testing"""
    
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
    """Deeper network for more complex testing"""
    
    def __init__(self, input_size: int = 784, hidden_sizes: list = None, num_classes: int = 10):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128, 64]
        
        layers = []
        layers.append(nn.Flatten())
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def create_test_model(model_type: str = "linear", **kwargs) -> nn.Module:
    """
    Factory function to create test models
    
    Args:
        model_type: Type of model ('linear', 'conv', 'tiny', 'deep')
        **kwargs: Additional arguments for model construction
        
    Returns:
        PyTorch model instance
    """
    model_map = {
        "linear": SimpleLinearNet,
        "conv": SimpleConvNet,
        "tiny": TinyNet,
        "deep": DeepLinearNet
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_map.keys())}")
    
    model_class = model_map[model_type]
    return model_class(**kwargs)

def initialize_model_weights(model: nn.Module, method: str = "xavier") -> nn.Module:
    """
    Initialize model weights using specified method
    
    Args:
        model: PyTorch model
        method: Initialization method ('xavier', 'kaiming', 'normal', 'zero')
        
    Returns:
        Model with initialized weights
    """
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if method == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif method == "kaiming":
                nn.init.kaiming_uniform_(m.weight)
            elif method == "normal":
                nn.init.normal_(m.weight, 0, 0.1)
            elif method == "zero":
                nn.init.zeros_(m.weight)
            
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        elif isinstance(m, nn.Conv2d):
            if method == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif method == "kaiming":
                nn.init.kaiming_uniform_(m.weight)
            elif method == "normal":
                nn.init.normal_(m.weight, 0, 0.1)
    
    model.apply(init_weights)
    return model

def load_pretrained_model(model_path: str) -> nn.Module:
    """
    Load a pretrained model from file
    
    Args:
        model_path: Path to model file
        
    Returns:
        Loaded PyTorch model
    """
    if model_path.endswith('.pt') or model_path.endswith('.pth'):
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        return model
    elif model_path.endswith('.onnx'):
        # For now, we'll implement ONNX loading later
        raise NotImplementedError("ONNX loading will be implemented in later steps")
    else:
        raise ValueError(f"Unsupported model format: {model_path}")

def create_sample_input(model_type: str, batch_size: int = 1) -> torch.Tensor:
    """
    Create sample input tensor for testing
    
    Args:
        model_type: Type of model the input is for
        batch_size: Batch size for input
        
    Returns:
        Sample input tensor
    """
    if model_type in ["linear", "deep"]:
        return torch.randn(batch_size, 1, 28, 28)  # MNIST-like
    elif model_type == "conv":
        return torch.randn(batch_size, 1, 28, 28)  # MNIST-like
    elif model_type == "tiny":
        return torch.randn(batch_size, 10)  # Simple vector
    else:
        return torch.randn(batch_size, 1, 28, 28)  # Default

# Predefined model configurations for common use cases
MODEL_CONFIGS = {
    "mnist_simple": {
        "type": "linear",
        "input_size": 784,
        "hidden_size": 100,
        "num_classes": 10
    },
    "mnist_conv": {
        "type": "conv",
        "num_classes": 10
    },
    "cifar_simple": {
        "type": "linear", 
        "input_size": 3072,  # 32*32*3
        "hidden_size": 256,
        "num_classes": 10
    },
    "toy_problem": {
        "type": "tiny",
        "input_size": 10,
        "num_classes": 3
    },
    "deep_network": {
        "type": "deep",
        "input_size": 784,
        "hidden_sizes": [512, 256, 128, 64],
        "num_classes": 10
    }
}

def create_model_from_config(config_name: str) -> nn.Module:
    """
    Create model from predefined configuration
    
    Args:
        config_name: Name of configuration from MODEL_CONFIGS
        
    Returns:
        Configured PyTorch model
    """
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[config_name].copy()
    model_type = config.pop("type")
    return create_test_model(model_type, **config)
