# src/core/verification/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import torch
import numpy as np
from enum import Enum

class VerificationStatus(Enum):
    """Status codes for verification results"""
    VERIFIED = "verified"
    FALSIFIED = "falsified" 
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
    ERROR = "error"

@dataclass
class VerificationResult:
    """Standard result format for all verification methods"""
    verified: bool
    status: VerificationStatus
    bounds: Optional[Dict[str, torch.Tensor]] = None
    counterexample: Optional[torch.Tensor] = None
    verification_time: Optional[float] = None
    memory_usage: Optional[float] = None
    additional_info: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate result consistency"""
        if self.verified and self.status not in [VerificationStatus.VERIFIED, VerificationStatus.UNKNOWN]:
            raise ValueError(f"Inconsistent result: verified=True but status={self.status}")
        if not self.verified and self.status == VerificationStatus.VERIFIED:
            raise ValueError(f"Inconsistent result: verified=False but status=VERIFIED")

@dataclass
class VerificationConfig:
    """Configuration for verification algorithms"""
    method: str = "alpha-beta-crown"
    timeout: int = 300  # seconds
    epsilon: float = 0.1
    norm: str = "inf"  # "inf", "2"
    batch_size: int = 1
    gpu_enabled: bool = False  # Set to False for CPU-only
    precision: str = "float32"
    
    # Advanced options
    bound_method: str = "CROWN"  # "IBP", "CROWN", "alpha-CROWN"
    optimization_steps: int = 20
    lr: float = 0.01
    
    def __post_init__(self):
        """Validate configuration"""
        if self.norm not in ["inf", "2"]:
            raise ValueError(f"Unsupported norm: {self.norm}")
        if self.epsilon <= 0:
            raise ValueError(f"Epsilon must be positive: {self.epsilon}")
        if self.timeout <= 0:
            raise ValueError(f"Timeout must be positive: {self.timeout}")
        if self.bound_method not in ["IBP", "CROWN", "alpha-CROWN"]:
            raise ValueError(f"Unsupported bound method: {self.bound_method}")

class VerificationEngine(ABC):
    """Base class for verification algorithm implementations"""
    
    @abstractmethod
    def verify(self, network: torch.nn.Module, input_sample: torch.Tensor, 
               config: VerificationConfig) -> VerificationResult:
        """
        Main verification entry point
        
        Args:
            network: PyTorch neural network to verify
            input_sample: Input tensor to verify around  
            config: Verification configuration
            
        Returns:
            VerificationResult with bounds and verification status
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return supported layers, activations, and specifications"""
        pass
    
    def verify_batch(self, network: torch.nn.Module, input_samples: torch.Tensor,
                    config: VerificationConfig) -> List[VerificationResult]:
        """
        Verify multiple inputs (default implementation - can be overridden)
        
        Args:
            network: PyTorch neural network
            input_samples: Batch of input tensors
            config: Verification configuration
            
        Returns:
            List of VerificationResult objects
        """
        results = []
        for i in range(input_samples.shape[0]):
            result = self.verify(network, input_samples[i:i+1], config)
            results.append(result)
        return results
    
    def get_device(self) -> torch.device:
        """Get the device this engine runs on"""
        return getattr(self, 'device', torch.device('cpu'))

class PropertySpecification:
    """Represents a property to be verified"""
    
    def __init__(self, property_type: str = "robustness", **kwargs):
        self.property_type = property_type
        self.parameters = kwargs
    
    def __repr__(self):
        return f"PropertySpecification(type={self.property_type}, params={self.parameters})"

@dataclass
class ModelInfo:
    """Information about the neural network being verified"""
    input_shape: tuple
    output_shape: tuple
    num_parameters: int
    layer_types: List[str]
    activation_functions: List[str]
    
    @classmethod
    def from_model(cls, model: torch.nn.Module, input_shape: tuple) -> 'ModelInfo':
        """Extract model information from a PyTorch model"""
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        # Get layer and activation types
        layer_types = []
        activation_functions = []
        
        for module in model.modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_type = type(module).__name__
                layer_types.append(layer_type)
                
                if layer_type in ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'ELU']:
                    activation_functions.append(layer_type)
        
        # Determine output shape by running a forward pass
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape)
            try:
                output = model(dummy_input)
                output_shape = tuple(output.shape[1:])  # Remove batch dimension
            except Exception:
                output_shape = (None,)  # Unknown output shape
        
        return cls(
            input_shape=input_shape,
            output_shape=output_shape,
            num_parameters=num_params,
            layer_types=list(set(layer_types)),
            activation_functions=list(set(activation_functions))
        )
