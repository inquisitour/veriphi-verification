# src/core/attacks/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
from enum import Enum
import os

class AttackStatus(Enum):
    """Status codes for attack results"""
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"

@dataclass
class AttackResult:
    """Result of an adversarial attack"""
    success: bool
    status: AttackStatus
    adversarial_example: Optional[torch.Tensor] = None
    original_prediction: Optional[int] = None
    adversarial_prediction: Optional[int] = None
    perturbation_norm: Optional[float] = None
    attack_time: Optional[float] = None
    iterations_used: Optional[int] = None
    confidence_original: Optional[float] = None
    confidence_adversarial: Optional[float] = None
    confidence_drop: Optional[float] = None
    additional_info: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate result consistency"""
        if self.success and self.status != AttackStatus.SUCCESS:
            raise ValueError(f"Inconsistent result: success=True but status={self.status}")
        if not self.success and self.status == AttackStatus.SUCCESS:
            raise ValueError(f"Inconsistent result: success=False but status=SUCCESS")

@dataclass
class AttackConfig:
    """Configuration for adversarial attacks"""
    epsilon: float = 0.1
    norm: str = "inf"  # "inf", "2", "1"
    targeted: bool = False
    target_class: Optional[int] = None
    max_iterations: int = 20
    step_size: Optional[float] = None
    clip_min: float = 0.0
    clip_max: float = 1.0
    
    # Advanced options
    random_start: bool = False
    early_stopping: bool = True
    confidence_threshold: float = 0.5
    loss_function: str = "cross_entropy"
    
    def __post_init__(self):
        """Validate configuration"""
        if self.norm not in ["inf", "2", "1"]:
            raise ValueError(f"Unsupported norm: {self.norm}")
        if self.epsilon <= 0:
            raise ValueError(f"Epsilon must be positive: {self.epsilon}")
        if self.max_iterations <= 0:
            raise ValueError(f"Max iterations must be positive: {self.max_iterations}")
        if self.targeted and self.target_class is None:
            raise ValueError("Target class must be specified for targeted attacks")
        if self.step_size is not None and self.step_size <= 0:
            raise ValueError(f'Step size must be positive: {self.step_size}')
        if self.clip_min >= self.clip_max:
            raise ValueError(f"clip_min ({self.clip_min}) must be < clip_max ({self.clip_max})")

class AdversarialAttack(ABC):
    """Base class for adversarial attacks"""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize attack
        
        Args:
            device: Device to run on ('cpu', 'cuda', or None for auto-detection)
        """
        if device is None:
            device = os.environ.get("VERIPHI_DEVICE", "cpu")
        self.device = torch.device(device)
    
    @abstractmethod
    def attack(self, model: torch.nn.Module, input_tensor: torch.Tensor, 
               config: AttackConfig) -> AttackResult:
        """
        Generate adversarial example
        
        Args:
            model: Target neural network
            input_tensor: Clean input to attack
            config: Attack configuration
            
        Returns:
            AttackResult with adversarial example and metadata
        """
        pass
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return attack capabilities and metadata"""
        return {
            'name': self.__class__.__name__,
            'norms': ['inf', '2'],
            'targeted': True,
            'iterative': False,
            'requires_gradients': True,
            'supports_batch': False
        }
    def _preprocess_input(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess input tensor"""
        if input_tensor.dim() < 2:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        return input_tensor
    
    def _get_predictions(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> tuple:
        """Get model predictions and confidence scores"""
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(output, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
        return predictions, confidence, output
    
    def _compute_perturbation_norm(self, original: torch.Tensor, perturbed: torch.Tensor, 
                                  norm: str) -> float:
        """Compute perturbation norm"""
        perturbation = perturbed - original
        if norm == 'inf':
            return torch.max(torch.abs(perturbation)).item()
        elif norm == '2':
            return torch.norm(perturbation.view(perturbation.shape[0], -1), dim=1).item()
        else:
            raise ValueError(f'Unsupported norm: {norm}')
    
    def _clamp_to_valid_range(self, tensor: torch.Tensor, 
                             min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
        """Clamp tensor values to valid range"""
        return torch.clamp(tensor, min_val, max_val)
    
    def _preprocess_input(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess input tensor for attack"""
        # Ensure batch dimension exists
        if input_tensor.dim() < 2:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        return input_tensor
    
    def _compute_perturbation_norm(self, original: torch.Tensor, 
                                 adversarial: torch.Tensor, norm: str) -> float:
        """Compute perturbation norm"""
        perturbation = adversarial - original
        
        if norm == "inf":
            return torch.max(torch.abs(perturbation)).item()
        elif norm == "2":
            return torch.norm(perturbation.view(perturbation.shape[0], -1), 
                            dim=1, p=2).item()
        elif norm == "1":
            return torch.norm(perturbation.view(perturbation.shape[0], -1), 
                            dim=1, p=1).item()
        else:
            raise ValueError(f"Unsupported norm: {norm}")
    
    def _project_perturbation(self, perturbation: torch.Tensor, 
                            epsilon: float, norm: str) -> torch.Tensor:
        """Project perturbation to satisfy norm constraint"""
        if norm == "inf":
            return torch.clamp(perturbation, -epsilon, epsilon)
        elif norm == "2":
            batch_size = perturbation.shape[0]
            perturbation_flat = perturbation.view(batch_size, -1)
            perturbation_norm = torch.norm(perturbation_flat, dim=1, p=2, keepdim=True)
            
            # Avoid division by zero
            perturbation_norm = torch.clamp(perturbation_norm, min=1e-8)
            
            # Scale down if norm exceeds epsilon
            scale = torch.min(torch.ones_like(perturbation_norm), 
                            epsilon / perturbation_norm)
            
            scaled_perturbation = perturbation_flat * scale
            return scaled_perturbation.view_as(perturbation)
        elif norm == "1":
            batch_size = perturbation.shape[0]
            perturbation_flat = perturbation.view(batch_size, -1)
            perturbation_norm = torch.norm(perturbation_flat, dim=1, p=1, keepdim=True)
            
            # Avoid division by zero
            perturbation_norm = torch.clamp(perturbation_norm, min=1e-8)
            
            # Scale down if norm exceeds epsilon
            scale = torch.min(torch.ones_like(perturbation_norm), 
                            epsilon / perturbation_norm)
            
            scaled_perturbation = perturbation_flat * scale
            return scaled_perturbation.view_as(perturbation)
        else:
            raise ValueError(f"Unsupported norm: {norm}")
    
    def _clip_adversarial(self, adversarial: torch.Tensor, 
                         config: AttackConfig) -> torch.Tensor:
        """Clip adversarial example to valid range"""
        return torch.clamp(adversarial, config.clip_min, config.clip_max)

class GradientBasedAttack(AdversarialAttack):
    """Base class for gradient-based attacks"""
    
    def _compute_loss(self, model: torch.nn.Module, input_tensor: torch.Tensor,
                     config: AttackConfig, original_prediction: int) -> torch.Tensor:
        """Compute loss for gradient computation"""
        output = model(input_tensor)
        
        if config.targeted and config.target_class is not None:
            # Targeted attack - minimize loss for target class
            target = torch.tensor([config.target_class], device=self.device)
            loss = -torch.nn.functional.cross_entropy(output, target)
        else:
            # Untargeted attack - maximize loss for true class
            true_label = torch.tensor([original_prediction], device=self.device)
            loss = torch.nn.functional.cross_entropy(output, true_label)
        
        return loss
    
    def _get_step_size(self, config: AttackConfig) -> float:
        """Get step size for iterative attacks"""
        if config.step_size is not None:
            return config.step_size
        
        # Default step sizes based on norm and epsilon
        if config.norm == "inf":
            return config.epsilon / 4.0
        elif config.norm == "2":
            return config.epsilon / 2.0
        else:
            return config.epsilon / 10.0

class SingleStepAttack(GradientBasedAttack):
    """Base class for single-step gradient attacks"""
    
    def get_capabilities(self) -> Dict[str, Any]:
        capabilities = super().get_capabilities()
        capabilities.update({
            'iterative': False,
            'single_step': True
        })
        return capabilities

class IterativeAttack(GradientBasedAttack):
    """Base class for iterative gradient attacks"""
    
    def get_capabilities(self) -> Dict[str, Any]:
        capabilities = super().get_capabilities()
        capabilities.update({
            'iterative': True,
            'supports_early_stopping': True
        })
        return capabilities
    
    def _check_early_stopping(self, model: torch.nn.Module, 
                            adversarial: torch.Tensor,
                            config: AttackConfig,
                            original_prediction: int) -> bool:
        """Check if attack should stop early"""
        if not config.early_stopping:
            return False
        
        with torch.no_grad():
            output = model(adversarial)
            prediction = torch.argmax(output, dim=1).item()
            
            if config.targeted and config.target_class is not None:
                # For targeted attacks, stop if we reach target
                return prediction == config.target_class
            else:
                # For untargeted attacks, stop if prediction changes
                return prediction != original_prediction


class TargetedAttackMixin:
    """Mixin for targeted attack functionality"""
    
    def _compute_targeted_loss(self, output: torch.Tensor, target_class: int, 
                              loss_function: str = "cross_entropy") -> torch.Tensor:
        """Compute loss for targeted attacks"""
        target = torch.tensor([target_class], device=output.device)
        
        if loss_function == "cross_entropy":
            # Minimize loss for target class (negative loss)
            return -torch.nn.functional.cross_entropy(output, target)
        elif loss_function == "margin":
            # Maximize margin for target class
            target_logit = output[0, target_class]
            other_logits = torch.cat([output[0, :target_class], output[0, target_class+1:]])
            max_other = torch.max(other_logits)
            return -(target_logit - max_other)
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")

class UntargetedAttackMixin:
    """Mixin for untargeted attack functionality"""
    
    def _compute_untargeted_loss(self, output: torch.Tensor, true_class: int,
                                loss_function: str = "cross_entropy") -> torch.Tensor:
        """Compute loss for untargeted attacks"""
        true_label = torch.tensor([true_class], device=output.device)
        
        if loss_function == "cross_entropy":
            # Maximize loss for true class
            return torch.nn.functional.cross_entropy(output, true_label)
        elif loss_function == "margin":
            # Minimize margin for true class
            true_logit = output[0, true_class]
            other_logits = torch.cat([output[0, :true_class], output[0, true_class+1:]])
            max_other = torch.max(other_logits)
            return -(max_other - true_logit)
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")

class AttackMetrics:
    """Utility class for computing attack metrics"""
    
    @staticmethod
    def success_rate(results: list) -> float:
        """Compute attack success rate"""
        if not results:
            return 0.0
        successful = sum(1 for r in results if r.success)
        return successful / len(results)
    
    @staticmethod
    def average_perturbation(results: list) -> float:
        """Compute average perturbation norm"""
        perturbations = [r.perturbation_norm for r in results if r.perturbation_norm is not None]
        if not perturbations:
            return 0.0
        return sum(perturbations) / len(perturbations)
    
    @staticmethod
    def average_iterations(results: list) -> float:
        """Compute average iterations used"""
        iterations = [r.iterations_used for r in results if r.iterations_used is not None]
        if not iterations:
            return 0.0
        return sum(iterations) / len(iterations)
    
    @staticmethod
    def confidence_drop(results: list) -> float:
        """Compute average confidence drop"""
        drops = []
        for r in results:
            if r.confidence_original is not None and r.confidence_adversarial is not None:
                drops.append(r.confidence_original - r.confidence_adversarial)
        
        if not drops:
            return 0.0
        return sum(drops) / len(drops)