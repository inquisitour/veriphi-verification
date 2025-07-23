# src/core/__init__.py
"""
Core verification and attack functionality for neural network robustness analysis.

This module now includes both formal verification and adversarial attacks for
comprehensive neural network security evaluation.
"""

import torch
from typing import Optional, List

# Verification components
from .verification import (
    VerificationEngine, 
    VerificationResult, 
    VerificationConfig, 
    VerificationStatus,
    AlphaBetaCrownEngine,
    PropertySpecification,
    ModelInfo
)

# Attack-guided verification
from .verification.attack_guided import (
    AttackGuidedEngine,
    create_attack_guided_engine,
    create_verification_engine_v2
)

# Attack components
from .attacks import (
    AdversarialAttack, 
    AttackResult, 
    AttackConfig, 
    AttackStatus, 
    FGSMAttack,
    IterativeFGSM,
    AttackMetrics,
    create_attack,
    list_available_attacks
)

# Model utilities
from .models import (
    create_test_model, 
    create_sample_input,
    create_model_from_config,
    MODEL_CONFIGS
)

class VeriphiCore:
    """
    Enhanced main interface for the verification tool with attack capabilities
    """
    
    def __init__(self, use_attacks: bool = True, device: str = 'cpu', 
                 attack_timeout: float = 10.0):
        """
        Initialize the core verification system
        
        Args:
            use_attacks: Whether to use attack-guided verification (recommended)
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            attack_timeout: Maximum time to spend on attacks before formal verification
        """
        self.use_attacks = use_attacks
        self.device = device
        self.attack_timeout = attack_timeout
        
        # Initialize appropriate engine
        if use_attacks:
            self.engine = create_attack_guided_engine(device, attack_timeout)
            print(f"✓ Attack-guided verification engine initialized")
        else:
            self.engine = AlphaBetaCrownEngine(device)
            print(f"✓ Formal verification engine initialized")
    
    def verify_robustness(self, model: torch.nn.Module, input_sample: torch.Tensor, 
                         epsilon: float = 0.1, norm: str = "inf", 
                         timeout: int = 300, bound_method: str = "CROWN") -> VerificationResult:
        """
        Main entry point for robustness verification
        
        Args:
            model: PyTorch neural network to verify
            input_sample: Input tensor to verify around
            epsilon: Perturbation bound
            norm: Norm type ("inf" or "2")
            timeout: Maximum verification time in seconds
            bound_method: Bound computation method ("IBP", "CROWN", "alpha-CROWN")
        
        Returns:
            VerificationResult with verification status and details
        """
        config = VerificationConfig(
            epsilon=epsilon,
            norm=norm,
            timeout=timeout,
            method="alpha-beta-crown",
            bound_method=bound_method
        )
        
        if hasattr(self.engine, 'verify_with_attacks'):
            return self.engine.verify_with_attacks(model, input_sample, config)
        else:
            return self.engine.verify(model, input_sample, config)
    
    def verify_batch(self, model: torch.nn.Module, input_samples: torch.Tensor,
                    epsilon: float = 0.1, norm: str = "inf", 
                    timeout: int = 300) -> List[VerificationResult]:
        """
        Verify multiple inputs at once
        
        Args:
            model: PyTorch neural network
            input_samples: Batch of input tensors
            epsilon: Perturbation bound
            norm: Norm type
            timeout: Total timeout for all samples
            
        Returns:
            List of VerificationResult objects
        """
        config = VerificationConfig(
            epsilon=epsilon,
            norm=norm,
            timeout=timeout // input_samples.shape[0],  # Divide timeout per sample
            method="alpha-beta-crown"
        )
        
        if hasattr(self.engine, 'verify_batch_with_attacks'):
            return self.engine.verify_batch_with_attacks(model, input_samples, config)
        else:
            return self.engine.verify_batch(model, input_samples, config)
    
    def attack_model(self, model: torch.nn.Module, input_sample: torch.Tensor,
                    attack_name: str = "fgsm", epsilon: float = 0.1, 
                    norm: str = "inf", targeted: bool = False, 
                    target_class: Optional[int] = None) -> AttackResult:
        """
        Attack model to generate adversarial examples
        
        Args:
            model: PyTorch neural network
            input_sample: Input tensor to perturb
            attack_name: Name of attack ("fgsm", "i-fgsm")
            epsilon: Perturbation bound
            norm: Norm type
            targeted: Whether to perform targeted attack
            target_class: Target class for targeted attacks
            
        Returns:
            AttackResult with adversarial example and metadata
        """
        attack = create_attack(attack_name, self.device)
        config = AttackConfig(
            epsilon=epsilon,
            norm=norm,
            targeted=targeted,
            target_class=target_class,
            max_iterations=20
        )
        
        return attack.attack(model, input_sample, config)
    
    def evaluate_robustness(self, model: torch.nn.Module, test_inputs: torch.Tensor,
                           epsilons: List[float] = [0.01, 0.05, 0.1, 0.2],
                           norm: str = "inf") -> dict:
        """
        Comprehensive robustness evaluation across multiple epsilon values
        
        Args:
            model: PyTorch neural network
            test_inputs: Test input samples
            epsilons: List of perturbation bounds to test
            norm: Norm type
            
        Returns:
            Dictionary with robustness statistics
        """
        results = {}
        
        print(f"🔍 Evaluating robustness across {len(epsilons)} epsilon values")
        print(f"   Test samples: {test_inputs.shape[0]}")
        print(f"   Epsilons: {epsilons}")
        
        for eps in epsilons:
            print(f"\n📊 Testing ε = {eps}")
            
            verification_results = self.verify_batch(
                model, test_inputs, epsilon=eps, norm=norm, timeout=60
            )
            
            # Compute statistics
            total_samples = len(verification_results)
            verified_count = sum(1 for r in verification_results if r.verified)
            falsified_count = sum(1 for r in verification_results if r.status == VerificationStatus.FALSIFIED)
            error_count = sum(1 for r in verification_results if r.status == VerificationStatus.ERROR)
            
            verification_rate = verified_count / total_samples
            falsification_rate = falsified_count / total_samples
            
            avg_time = sum(r.verification_time or 0 for r in verification_results) / total_samples
            
            results[eps] = {
                'verification_rate': verification_rate,
                'falsification_rate': falsification_rate,
                'error_rate': error_count / total_samples,
                'verified_count': verified_count,
                'falsified_count': falsified_count,
                'error_count': error_count,
                'total_samples': total_samples,
                'average_time': avg_time,
                'results': verification_results
            }
            
            print(f"   Verified: {verified_count}/{total_samples} ({verification_rate:.1%})")
            print(f"   Falsified: {falsified_count}/{total_samples} ({falsification_rate:.1%})")
            print(f"   Average time: {avg_time:.3f}s")
        
        return results
    
    def get_capabilities(self) -> dict:
        """Get capabilities of the verification system"""
        capabilities = self.engine.get_capabilities()
        capabilities.update({
            'attack_support': self.use_attacks,
            'available_attacks': list_available_attacks(),
            'batch_verification': True,
            'robustness_evaluation': True
        })
        return capabilities
    
    def configure_attacks(self, attack_names: List[str], attack_timeout: Optional[float] = None):
        """Configure which attacks to use (only for attack-guided engines)"""
        if hasattr(self.engine, 'configure_attacks'):
            self.engine.configure_attacks(attack_names, attack_timeout)
        else:
            print("Warning: Current engine does not support attack configuration")

# Convenience functions
def create_core_system(use_attacks: bool = True, device: str = 'cpu') -> VeriphiCore:
    """Create the core verification system"""
    return VeriphiCore(use_attacks=use_attacks, device=device)

def quick_robustness_check(model: torch.nn.Module, input_sample: torch.Tensor,
                          epsilon: float = 0.1) -> bool:
    """
    Quick robustness check - returns True if robust, False otherwise
    
    Args:
        model: PyTorch model
        input_sample: Input to check
        epsilon: Perturbation bound
        
    Returns:
        True if robust, False if falsified
    """
    core = create_core_system(use_attacks=True, device='cpu')
    result = core.verify_robustness(model, input_sample, epsilon=epsilon, timeout=30)
    return result.verified

__all__ = [
    # Main interface
    'VeriphiCore',
    'create_core_system',
    'quick_robustness_check',
    
    # Verification components
    'VerificationEngine', 
    'VerificationResult', 
    'VerificationConfig', 
    'VerificationStatus',
    'AlphaBetaCrownEngine',
    'AttackGuidedEngine',
    'PropertySpecification',
    'ModelInfo',
    
    # Attack components  
    'AdversarialAttack',
    'AttackResult', 
    'AttackConfig', 
    'AttackStatus',
    'FGSMAttack',
    'IterativeFGSM',
    'AttackMetrics',
    'create_attack',
    'list_available_attacks',
    
    # Model utilities
    'create_test_model',
    'create_sample_input',
    'create_model_from_config',
    'MODEL_CONFIGS',
    
    # Factory functions
    'create_attack_guided_engine',
    'create_verification_engine_v2'
]