# src/core/attacks/fgsm.py
import torch
import torch.nn.functional as F
import time
from typing import Optional

from .base import AdversarialAttack, AttackResult, AttackConfig, AttackStatus, TargetedAttackMixin, UntargetedAttackMixin

class FGSMAttack(AdversarialAttack, TargetedAttackMixin, UntargetedAttackMixin):
    """Fast Gradient Sign Method (FGSM) Attack"""
    
    def __init__(self, device: Optional[str] = None):
        super().__init__(device)
        print(f"Initialized FGSM attack on device: {self.device}")
    
    def attack(self, model: torch.nn.Module, input_tensor: torch.Tensor, 
               config: AttackConfig) -> AttackResult:
        start_time = time.time()
        
        try:
            # Setup
            model = model.to(self.device).eval()
            
            # Preprocess input but don't require gradients yet
            if input_tensor.dim() < 2:
                input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            # Get original predictions (no gradients needed)
            with torch.no_grad():
                original_output = model(input_tensor)
                original_pred = torch.argmax(original_output, dim=1)
                original_conf = torch.softmax(original_output, dim=1).max(dim=1)[0]
            
            original_class = original_pred[0].item()
            print(f'FGSM Attack - Original: class {original_class} (conf: {original_conf[0]:.3f})')
            
            # NOW enable gradients for the attack
            input_tensor = input_tensor.clone().detach().requires_grad_(True)
            
            # Forward pass with gradients
            output = model(input_tensor)
            
            # Compute loss
            if config.targeted and config.target_class is not None:
                loss = self._compute_targeted_loss(output, config.target_class, config.loss_function)
                print(f'Targeted attack: {original_class} -> {config.target_class}')
            else:
                loss = self._compute_untargeted_loss(output, original_class, config.loss_function)
                print(f'Untargeted attack: trying to fool class {original_class}')
            
            # Compute gradients
            model.zero_grad()
            loss.backward()
            
            # Get gradients
            data_grad = input_tensor.grad.data
            
            # Generate adversarial example
            if config.norm == "inf":
                sign_data_grad = data_grad.sign()
                perturbed_data = input_tensor + config.epsilon * sign_data_grad
            elif config.norm == "2":
                grad_norm = torch.norm(data_grad.view(data_grad.shape[0], -1), dim=1, keepdim=True)
                grad_norm = torch.clamp(grad_norm, min=1e-8)
                normalized_grad = data_grad / grad_norm.view(-1, 1, 1, 1) if data_grad.dim() == 4 else data_grad / grad_norm
                perturbed_data = input_tensor + config.epsilon * normalized_grad
            
            # Clamp to valid range
            perturbed_data = torch.clamp(perturbed_data, 0, 1)
            
            # Get adversarial predictions
            with torch.no_grad():
                adv_output = model(perturbed_data)
                adv_pred = torch.argmax(adv_output, dim=1)
                adv_conf = torch.softmax(adv_output, dim=1).max(dim=1)[0]
            
            adv_class = adv_pred[0].item()
            
            # Determine success
            if config.targeted and config.target_class is not None:
                success = (adv_class == config.target_class)
            else:
                success = (adv_class != original_class)
            
            # Compute perturbation norm
            perturbation = perturbed_data - input_tensor.detach()
            if config.norm == "inf":
                pert_norm = torch.max(torch.abs(perturbation)).item()
            else:
                pert_norm = torch.norm(perturbation.view(perturbation.shape[0], -1), dim=1).item()
            
            attack_time = time.time() - start_time
            
            print(f'Attack result: {"SUCCESS" if success else "FAILED"}')
            if success:
                print(f'Prediction change: {original_class} -> {adv_class}')
            
            return AttackResult(
                success=success,
                status=AttackStatus.SUCCESS if success else AttackStatus.FAILED,
                adversarial_example=perturbed_data.detach().cpu(),
                original_prediction=original_class,
                adversarial_prediction=adv_class,
                perturbation_norm=pert_norm,
                attack_time=attack_time,
                iterations_used=1,
                confidence_original=original_conf[0].item(),
                confidence_adversarial=adv_conf[0].item(),
                additional_info={
                    "epsilon": config.epsilon,
                    "norm": config.norm,
                    "targeted": config.targeted,
                    "target_class": config.target_class,
                    "loss_function": config.loss_function,
                    "confidence_drop": original_conf[0].item() - adv_conf[0].item()
                }
            )
            
        except Exception as e:
            attack_time = time.time() - start_time
            print(f'FGSM attack failed: {e}')
            return AttackResult(
                success=False,
                status=AttackStatus.ERROR,
                attack_time=attack_time,
                additional_info={"error": str(e), "error_type": type(e).__name__}
            )
            
        except Exception as e:
            attack_time = time.time() - start_time
            print(f"FGSM attack failed: {e}")
            return AttackResult(
                success=False,
                status=AttackStatus.ERROR,
                attack_time=attack_time,
                additional_info={"error": str(e), "error_type": type(e).__name__}
            )
    
    def get_capabilities(self) -> dict:
        """Return FGSM capabilities"""
        capabilities = super().get_capabilities()
        capabilities.update({
            'iterative': False,
            'single_step': True,
            'gradient_based': True,
            'supports_targeted': True,
            'supports_untargeted': True,
            'fast_execution': True
        })
        return capabilities

class IterativeFGSM(AdversarialAttack, TargetedAttackMixin, UntargetedAttackMixin):
    """Iterative FGSM (I-FGSM) - Basic Iterative Method"""
    
    def __init__(self, device: Optional[str] = None):
        super().__init__(device)
        print(f"Initialized I-FGSM attack on device: {self.device}")
    
    def attack(self, model: torch.nn.Module, input_tensor: torch.Tensor, 
               config: AttackConfig) -> AttackResult:
        start_time = time.time()
        
        try:
            # Setup
            model = model.to(self.device).eval()
            
            # Preprocess input but don't require gradients yet
            if input_tensor.dim() < 2:
                input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            # Get original predictions (no gradients needed)
            with torch.no_grad():
                original_output = model(input_tensor)
                original_pred = torch.argmax(original_output, dim=1)
                original_conf = torch.softmax(original_output, dim=1).max(dim=1)[0]
            
            original_class = original_pred[0].item()
            print(f'FGSM Attack - Original: class {original_class} (conf: {original_conf[0]:.3f})')
            
            # NOW enable gradients for the attack
            input_tensor = input_tensor.clone().detach().requires_grad_(True)
            
            # Forward pass with gradients
            output = model(input_tensor)
            
            # Compute loss
            if config.targeted and config.target_class is not None:
                loss = self._compute_targeted_loss(output, config.target_class, config.loss_function)
                print(f'Targeted attack: {original_class} -> {config.target_class}')
            else:
                loss = self._compute_untargeted_loss(output, original_class, config.loss_function)
                print(f'Untargeted attack: trying to fool class {original_class}')
            
            # Compute gradients
            model.zero_grad()
            loss.backward()
            
            # Get gradients
            data_grad = input_tensor.grad.data
            
            # Generate adversarial example
            if config.norm == "inf":
                sign_data_grad = data_grad.sign()
                perturbed_data = input_tensor + config.epsilon * sign_data_grad
            elif config.norm == "2":
                grad_norm = torch.norm(data_grad.view(data_grad.shape[0], -1), dim=1, keepdim=True)
                grad_norm = torch.clamp(grad_norm, min=1e-8)
                normalized_grad = data_grad / grad_norm.view(-1, 1, 1, 1) if data_grad.dim() == 4 else data_grad / grad_norm
                perturbed_data = input_tensor + config.epsilon * normalized_grad
            
            # Clamp to valid range
            perturbed_data = torch.clamp(perturbed_data, 0, 1)
            
            # Get adversarial predictions
            with torch.no_grad():
                adv_output = model(perturbed_data)
                adv_pred = torch.argmax(adv_output, dim=1)
                adv_conf = torch.softmax(adv_output, dim=1).max(dim=1)[0]
            
            adv_class = adv_pred[0].item()
            
            # Determine success
            if config.targeted and config.target_class is not None:
                success = (adv_class == config.target_class)
            else:
                success = (adv_class != original_class)
            
            # Compute perturbation norm
            perturbation = perturbed_data - input_tensor.detach()
            if config.norm == "inf":
                pert_norm = torch.max(torch.abs(perturbation)).item()
            else:
                pert_norm = torch.norm(perturbation.view(perturbation.shape[0], -1), dim=1).item()
            
            attack_time = time.time() - start_time
            
            print(f'Attack result: {"SUCCESS" if success else "FAILED"}')
            if success:
                print(f'Prediction change: {original_class} -> {adv_class}')
            
            return AttackResult(
                success=success,
                status=AttackStatus.SUCCESS if success else AttackStatus.FAILED,
                adversarial_example=perturbed_data.detach().cpu(),
                original_prediction=original_class,
                adversarial_prediction=adv_class,
                perturbation_norm=pert_norm,
                attack_time=attack_time,
                iterations_used=1,
                confidence_original=original_conf[0].item(),
                confidence_adversarial=adv_conf[0].item(),
                additional_info={
                    "epsilon": config.epsilon,
                    "norm": config.norm,
                    "targeted": config.targeted,
                    "target_class": config.target_class,
                    "loss_function": config.loss_function,
                    "confidence_drop": original_conf[0].item() - adv_conf[0].item()
                }
            )
            
        except Exception as e:
            attack_time = time.time() - start_time
            print(f'FGSM attack failed: {e}')
            return AttackResult(
                success=False,
                status=AttackStatus.ERROR,
                attack_time=attack_time,
                additional_info={"error": str(e), "error_type": type(e).__name__}
            )
            
        except Exception as e:
            attack_time = time.time() - start_time
            print(f"I-FGSM attack failed: {e}")
            return AttackResult(
                success=False,
                status=AttackStatus.ERROR,
                attack_time=attack_time,
                additional_info={"error": str(e), "error_type": type(e).__name__}
            )
    
    def get_capabilities(self) -> dict:
        """Return I-FGSM capabilities"""
        capabilities = super().get_capabilities()
        capabilities.update({
            'iterative': True,
            'single_step': False,
            'gradient_based': True,
            'supports_targeted': True,
            'supports_untargeted': True,
            'supports_random_start': True,
            'supports_early_stopping': True
        })
        return capabilities