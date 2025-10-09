# src/core/verification/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import torch
import numpy as np
from enum import Enum
import os

# -------------------------------------------------------------
# Global default device
# -------------------------------------------------------------
DEFAULT_DEVICE = os.environ.get("VERIPHI_DEVICE", "cpu")


# -------------------------------------------------------------
# Status and results
# -------------------------------------------------------------
class VerificationStatus(Enum):
    """Status codes for verification results."""
    VERIFIED = "verified"
    FALSIFIED = "falsified"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass
class VerificationResult:
    """Standard result format for all verification methods."""
    verified: bool
    status: VerificationStatus
    bounds: Optional[Dict[str, torch.Tensor]] = None
    counterexample: Optional[torch.Tensor] = None
    verification_time: Optional[float] = None
    memory_usage: Optional[float] = None
    additional_info: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate result consistency."""
        if self.verified and self.status not in [
            VerificationStatus.VERIFIED,
            VerificationStatus.UNKNOWN,
        ]:
            raise ValueError(
                f"Inconsistent result: verified=True but status={self.status}"
            )
        if not self.verified and self.status == VerificationStatus.VERIFIED:
            raise ValueError(
                f"Inconsistent result: verified=False but status=VERIFIED"
            )


# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
@dataclass
class VerificationConfig:
    """Configuration for verification algorithms."""
    method: str = "alpha-beta-crown"
    timeout: int = 300  # seconds
    epsilon: float = 0.1
    norm: str = "inf"  # "inf", "2"
    batch_size: int = 1
    gpu_enabled: bool = False
    precision: str = "float32"

    # Advanced options
    bound_method: str = "beta-CROWN"  # "IBP", "alpha-CROWN", "beta-CROWN"
    optimization_steps: int = 50
    lr: float = 0.005

    def __post_init__(self):
        """Validate configuration."""
        if self.norm not in ["inf", "2"]:
            raise ValueError(f"Unsupported norm: {self.norm}")
        if self.epsilon <= 0:
            raise ValueError(f"Epsilon must be positive: {self.epsilon}")
        if self.timeout <= 0:
            raise ValueError(f"Timeout must be positive: {self.timeout}")
        if self.bound_method not in ["IBP", "CROWN", "alpha-CROWN", "beta-CROWN"]:
            raise ValueError(f"Unsupported bound method: {self.bound_method}")


# -------------------------------------------------------------
# Abstract engine base
# -------------------------------------------------------------
class VerificationEngine(ABC):
    """Base class for verification algorithm implementations."""

    @abstractmethod
    def verify(
        self,
        network: torch.nn.Module,
        input_sample: torch.Tensor,
        config: VerificationConfig,
    ) -> VerificationResult:
        """Main verification entry point."""
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return supported layers, activations, and specifications."""
        pass

    # ---------------------------------------------------------
    def verify_batch(
        self,
        network: torch.nn.Module,
        input_samples: torch.Tensor,
        config: VerificationConfig,
    ) -> List[VerificationResult]:
        """Default batch verification (sequential)."""
        results = []
        for i in range(input_samples.shape[0]):
            result = self.verify(network, input_samples[i : i + 1], config)
            results.append(result)
        return results

    # ---------------------------------------------------------
    def get_device(self) -> torch.device:
        """Return the device this engine runs on (default: VERIPHI_DEVICE)."""
        return getattr(self, "device", torch.device(DEFAULT_DEVICE))


# -------------------------------------------------------------
# Specification and model metadata
# -------------------------------------------------------------
class PropertySpecification:
    """Represents a property to be verified (e.g., robustness)."""

    def __init__(self, property_type: str = "robustness", **kwargs):
        self.property_type = property_type
        self.parameters = kwargs

    def __repr__(self):
        return f"PropertySpecification(type={self.property_type}, params={self.parameters})"


@dataclass
class ModelInfo:
    """Information about the neural network being verified."""
    input_shape: tuple
    output_shape: tuple
    num_parameters: int
    layer_types: List[str]
    activation_functions: List[str]

    @classmethod
    def from_model(cls, model: torch.nn.Module, input_shape: tuple) -> ModelInfo:
        """Extract model information from a PyTorch model."""
        device = torch.device(os.environ.get("VERIPHI_DEVICE", "cpu"))

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())

        # Collect leaf layer and activation names
        layer_types, activations = [], []
        for module in model.modules():
            if len(list(module.children())) == 0:
                layer_type = type(module).__name__
                layer_types.append(layer_type)
                if layer_type in ["ReLU", "Sigmoid", "Tanh", "LeakyReLU", "ELU"]:
                    activations.append(layer_type)

        # Determine output shape with dummy forward
        model = model.to(device).eval()
        dummy_input = torch.randn(1, *input_shape, device=device)
        with torch.no_grad():
            try:
                output = model(dummy_input)
                out_shape = tuple(output.shape[1:])
            except Exception:
                out_shape = (None,)

        return cls(
            input_shape=input_shape,
            output_shape=out_shape,
            num_parameters=num_params,
            layer_types=sorted(set(layer_types)),
            activation_functions=sorted(set(activations)),
        )
