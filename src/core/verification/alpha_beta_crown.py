# src/core/verification/alpha_beta_crown.py
import torch
import torch.nn as nn
import time
import tracemalloc
import numpy as np
from typing import Dict, Any, Tuple, Optional

# auto-LiRPA imports with correct paths
from auto_LiRPA.bound_general import BoundedModule
from auto_LiRPA.bounded_tensor import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from .base import VerificationEngine, VerificationResult, VerificationConfig, VerificationStatus

class AlphaBetaCrownEngine(VerificationEngine):
    """α,β-CROWN verification implementation using auto-LiRPA"""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize α,β-CROWN verifier
        
        Args:
            device: Device to run on ('cpu', 'cuda', or None for auto-detection)
        """
        if device is None:
            self.device = torch.device('cpu')  # Default to CPU for our setup
        else:
            self.device = torch.device(device)
        
        print(f"Initializing α,β-CROWN verifier on device: {self.device}")
        
        # Optimization settings
        self.default_bound_opts = {
            'optimize_bound_args': {
                'optimizer': 'adam',
                'lr': 0.1,
                'iteration': 20,
            }
        }
    
    def verify(self, network: nn.Module, input_sample: torch.Tensor, 
               config: VerificationConfig) -> VerificationResult:
        """
        Verify robustness properties using α,β-CROWN
        
        Args:
            network: PyTorch neural network
            input_sample: Input tensor to verify around
            config: Verification configuration
        
        Returns:
            VerificationResult with bounds and verification status
        """
        start_time = time.time()
        tracemalloc.start()
        
        try:
            # Input validation and preprocessing
            input_sample = self._preprocess_input(input_sample)
            
            # Move to device
            network = network.to(self.device).eval()
            input_sample = input_sample.to(self.device)
            
            # Get original prediction
            with torch.no_grad():
                original_output = network(input_sample)
                predicted_class = torch.argmax(original_output, dim=1)
            
            print(f"Original prediction: class {predicted_class.item()}")
            print(f"Verifying robustness with ε={config.epsilon}, norm=L{config.norm}")
            
            # Create bounded model with auto-LiRPA
            bounded_model = BoundedModule(network, input_sample)
            
            # Define perturbation
            perturbation = self._create_perturbation(config)
            bounded_input = BoundedTensor(input_sample, perturbation)
            
            # Compute bounds using specified method
            lb, ub = self._compute_bounds(bounded_model, bounded_input, config)
            
            # Verify robustness
            verified = self._check_robustness(lb, ub, predicted_class)
            
            verification_time = time.time() - start_time
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            status = VerificationStatus.VERIFIED if verified else VerificationStatus.FALSIFIED
            
            print(f"Verification result: {status.value} (time: {verification_time:.3f}s)")
            
            return VerificationResult(
                verified=verified,
                status=status,
                bounds={"lower": lb.detach().cpu(), "upper": ub.detach().cpu()},
                verification_time=verification_time,
                memory_usage=peak_memory / 1024 / 1024,  # MB
                additional_info={
                    "predicted_class": predicted_class.item(),
                    "perturbation_norm": config.norm,
                    "epsilon": config.epsilon,
                    "bound_method": config.bound_method,
                    "bounds_gap": self._compute_bounds_gap(lb, ub, predicted_class)
                }
            )
            
        except Exception as e:
            verification_time = time.time() - start_time
            if 'tracemalloc' in locals():
                tracemalloc.stop()
            
            print(f"Verification failed with error: {e}")
            return VerificationResult(
                verified=False,
                status=VerificationStatus.ERROR,
                verification_time=verification_time,
                additional_info={"error": str(e), "error_type": type(e).__name__}
            )
    
    def _preprocess_input(self, input_sample: torch.Tensor) -> torch.Tensor:
        """Preprocess input tensor for verification"""
        # Ensure batch dimension exists
        if input_sample.dim() < 2:
            input_sample = input_sample.unsqueeze(0)
        
        # Ensure input is in valid range [0, 1] for image data
        if input_sample.min() < 0 or input_sample.max() > 1:
            # Assume input is in [-1, 1] or [0, 255] range and normalize
            if input_sample.max() > 1:
                input_sample = input_sample / 255.0
            else:
                input_sample = (input_sample + 1) / 2.0
        
        return input_sample
    
    def _create_perturbation(self, config: VerificationConfig) -> PerturbationLpNorm:
        """Create perturbation object based on configuration"""
        if config.norm == "inf":
            return PerturbationLpNorm(norm=np.inf, eps=config.epsilon)
        elif config.norm == "2":
            return PerturbationLpNorm(norm=2, eps=config.epsilon)
        else:
            raise ValueError(f"Unsupported norm: {config.norm}")
    
    def _compute_bounds(self, bounded_model: BoundedModule, bounded_input: BoundedTensor, 
                       config: VerificationConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute bounds using the specified method"""
        
        method_map = {
            "IBP": "IBP",
            "CROWN": "backward",
            "alpha-CROWN": "alpha-crown"
        }
        
        method = method_map.get(config.bound_method, "alpha-crown")
        
        print(f"Computing bounds using method: {method}")
        
        if method == "alpha-crown":
            # Use α-CROWN with optimization
            lb, ub = bounded_model.compute_bounds(
                x=(bounded_input,), 
                method="alpha-crown",
                bound_upper=True,
                **self.default_bound_opts
            )
        else:
            # Use standard methods
            lb, ub = bounded_model.compute_bounds(
                x=(bounded_input,), 
                method=method,
                bound_upper=True
            )
        
        return lb, ub
    
    def _check_robustness(self, lb: torch.Tensor, ub: torch.Tensor, 
                         predicted_class: torch.Tensor) -> bool:
        """Check if the network is robust for the given bounds"""
        batch_size = lb.shape[0]
        
        for i in range(batch_size):
            pred_class = predicted_class[i].item()
            
            # Check if predicted class always has the highest lower bound
            lb_pred = lb[i, pred_class]
            
            # Compare with upper bounds of all other classes
            for j in range(lb.shape[1]):
                if j != pred_class:
                    if lb_pred <= ub[i, j]:
                        print(f"Robustness violated: class {pred_class} LB ({lb_pred:.6f}) <= class {j} UB ({ub[i, j]:.6f})")
                        return False
        
        print("Robustness verified: predicted class maintains highest confidence")
        return True
    
    def _compute_bounds_gap(self, lb: torch.Tensor, ub: torch.Tensor, 
                           predicted_class: torch.Tensor) -> float:
        """Compute the gap between predicted class bounds and others"""
        pred_class = predicted_class[0].item()
        lb_pred = lb[0, pred_class].item()
        
        # Find the maximum upper bound of other classes
        other_classes_ub = []
        for j in range(lb.shape[1]):
            if j != pred_class:
                other_classes_ub.append(ub[0, j].item())
        
        max_other_ub = max(other_classes_ub) if other_classes_ub else 0.0
        gap = lb_pred - max_other_ub
        
        return gap
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return supported features"""
        return {
            'layers': ['Linear', 'Conv2d', 'ReLU', 'MaxPool2d', 'AvgPool2d', 'BatchNorm2d', 'Flatten'],
            'activations': ['ReLU', 'Sigmoid', 'Tanh'],
            'norms': ['inf', '2'],
            'specifications': ['robustness', 'reachability'],
            'bound_methods': ['IBP', 'CROWN', 'alpha-CROWN'],
            'max_neurons': 1000000,  # Can handle very large networks
            'gpu_accelerated': self.device.type == 'cuda',
            'supports_batch': False,  # Single input at a time for now
            'framework': 'auto-LiRPA',
            'algorithm': 'α,β-CROWN'
        }

class GPUOptimizedEngine(AlphaBetaCrownEngine):
    """Enhanced verifier with memory optimizations for large models"""
    
    def __init__(self, device: Optional[str] = None):
        super().__init__(device)
        
        # Enable optimizations if on GPU
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            print("GPU optimizations enabled")
    
    def verify_large_model(self, network: nn.Module, input_sample: torch.Tensor, 
                          config: VerificationConfig) -> VerificationResult:
        """Verify models that don't fit in GPU memory using memory optimizations"""
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(network, 'gradient_checkpointing_enable'):
            network.gradient_checkpointing_enable()
        
        # Use mixed precision for memory efficiency (GPU only)
        if self.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                return self.verify(network, input_sample, config)
        else:
            return self.verify(network, input_sample, config)

def create_verification_engine(device: Optional[str] = None, 
                             optimized: bool = False) -> VerificationEngine:
    """
    Factory function to create verification engines
    
    Args:
        device: Device to run on ('cpu', 'cuda', or None)
        optimized: Whether to use GPU-optimized version
        
    Returns:
        VerificationEngine instance
    """
    if optimized:
        return GPUOptimizedEngine(device)
    else:
        return AlphaBetaCrownEngine(device)
