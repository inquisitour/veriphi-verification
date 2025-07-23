# src/core/verification/attack_guided.py
import torch
import time
from typing import List, Optional, Dict, Any

from .alpha_beta_crown import AlphaBetaCrownEngine
from .base import VerificationResult, VerificationConfig, VerificationStatus
from ..attacks import FGSMAttack, IterativeFGSM, AttackConfig, AttackStatus

class AttackGuidedEngine:
    """
    Attack-Guided Verification Engine
    
    This engine combines adversarial attacks with formal verification for maximum efficiency.
    The strategy is:
    1. Try fast attacks first to find counterexamples quickly
    2. If attacks fail, proceed with formal verification
    3. If formal verification times out, provide attack-based bounds
    """
    
    def __init__(self, device: Optional[str] = None, attack_timeout: float = 10.0):
        """
        Initialize attack-guided verification engine
        
        Args:
            device: Device to run on ('cpu', 'cuda', or None)
            attack_timeout: Maximum time to spend on attacks before formal verification
        """
        self.device = device if device else 'cpu'
        self.attack_timeout = attack_timeout
        
        # Initialize verification engine
        self.verifier = AlphaBetaCrownEngine(device)
        
        # Initialize attacks (ordered by speed vs. effectiveness)
        self.attacks = [
            FGSMAttack(device),           # Fastest
            IterativeFGSM(device)         # More thorough
        ]
        
        print(f"Initialized Attack-Guided Verification Engine on {self.device}")
        print(f"Available attacks: {[a.__class__.__name__ for a in self.attacks]}")
    
    def verify_with_attacks(self, network: torch.nn.Module, input_sample: torch.Tensor, 
                           config: VerificationConfig) -> VerificationResult:
        """
        Main verification method using attack-guided approach
        
        Args:
            network: Neural network to verify
            input_sample: Input to verify around
            config: Verification configuration
            
        Returns:
            VerificationResult with attack or verification outcome
        """
        total_start_time = time.time()
        
        print(f"🚀 Starting attack-guided verification")
        print(f"   Property: ε={config.epsilon}, norm=L{config.norm}")
        print(f"   Attack timeout: {self.attack_timeout}s")
        print(f"   Verification timeout: {config.timeout}s")
        
        # Phase 1: Attack phase - try to falsify quickly
        attack_result = self._attack_phase(network, input_sample, config)
        
        if attack_result.status == VerificationStatus.FALSIFIED:
            # Attack found counterexample - we're done!
            total_time = time.time() - total_start_time
            attack_result.verification_time = total_time
            attack_result.additional_info = attack_result.additional_info or {}
            attack_result.additional_info.update({
                "verification_method": "attack-guided",
                "phase_completed": "attack",
                "formal_verification_skipped": True
            })
            return attack_result
        
        # Phase 2: Formal verification phase
        print(f"   ⚡ Attacks completed, proceeding with formal verification...")
        
        # Adjust timeout for formal verification (subtract attack time)
        attack_time = time.time() - total_start_time
        remaining_timeout = max(10, config.timeout - attack_time)  # At least 10s for formal verification
        
        formal_config = VerificationConfig(
            method=config.method,
            timeout=int(remaining_timeout),
            epsilon=config.epsilon,
            norm=config.norm,
            batch_size=config.batch_size,
            gpu_enabled=config.gpu_enabled,
            precision=config.precision,
            bound_method=config.bound_method,
            optimization_steps=config.optimization_steps,
            lr=config.lr
        )
        
        formal_result = self.verifier.verify(network, input_sample, formal_config)
        
        # Combine results
        total_time = time.time() - total_start_time
        formal_result.verification_time = total_time
        
        if formal_result.additional_info is None:
            formal_result.additional_info = {}
        
        formal_result.additional_info.update({
            "verification_method": "attack-guided",
            "attack_phase_time": attack_time,
            "formal_phase_time": formal_result.verification_time - attack_time,
            "attacks_tried": [a.__class__.__name__ for a in self.attacks],
            "attack_phase_result": "no_counterexample_found"
        })
        
        return formal_result
    
    def _attack_phase(self, network: torch.nn.Module, input_sample: torch.Tensor, 
                     config: VerificationConfig) -> VerificationResult:
        """
        Attack phase: try multiple attacks to find counterexamples quickly
        
        Args:
            network: Neural network
            input_sample: Input sample
            config: Verification configuration
            
        Returns:
            VerificationResult (falsified if attack succeeds, unknown otherwise)
        """
        print(f"   🗡️ Phase 1: Attack phase (timeout: {self.attack_timeout}s)")
        
        attack_start_time = time.time()
        attack_results = []
        
        # Create attack configuration
        attack_config = AttackConfig(
            epsilon=config.epsilon,
            norm=config.norm,
            targeted=False,  # Untargeted attacks for robustness
            max_iterations=min(20, int(self.attack_timeout * 10)),  # Scale iterations with timeout
            early_stopping=True,
            random_start=False
        )
        
        for i, attack in enumerate(self.attacks):
            elapsed = time.time() - attack_start_time
            if elapsed >= self.attack_timeout:
                print(f"      ⏰ Attack timeout reached, stopping attack phase")
                break
            
            print(f"      Trying {attack.__class__.__name__}...")
            
            try:
                # Adjust remaining time for this attack
                remaining_time = self.attack_timeout - elapsed
                if hasattr(attack_config, 'max_time'):
                    attack_config.max_time = remaining_time
                
                attack_result = attack.attack(network, input_sample, attack_config)
                attack_results.append(attack_result)
                
                if attack_result.success:
                    print(f"      ✗ Counterexample found with {attack.__class__.__name__}!")
                    print(f"      Attack time: {attack_result.attack_time:.3f}s")
                    print(f"      Perturbation norm: {attack_result.perturbation_norm:.6f}")
                    
                    # Return falsified result
                    return VerificationResult(
                        verified=False,
                        status=VerificationStatus.FALSIFIED,
                        counterexample=attack_result.adversarial_example,
                        verification_time=time.time() - attack_start_time,
                        additional_info={
                            "falsification_method": attack.__class__.__name__,
                            "original_prediction": attack_result.original_prediction,
                            "adversarial_prediction": attack_result.adversarial_prediction,
                            "perturbation_norm": attack_result.perturbation_norm,
                            "confidence_drop": attack_result.additional_info.get("confidence_drop", 0),
                            "attack_details": attack_result.additional_info
                        }
                    )
                else:
                    print(f"      ○ {attack.__class__.__name__} failed to find counterexample")
                    
            except Exception as e:
                print(f"      ✗ {attack.__class__.__name__} error: {e}")
                continue
        
        attack_phase_time = time.time() - attack_start_time
        print(f"   ○ Attack phase completed ({attack_phase_time:.3f}s) - No counterexamples found")
        
        # Return unknown status (attacks failed)
        return VerificationResult(
            verified=False,  # We don't know yet
            status=VerificationStatus.UNKNOWN,
            verification_time=attack_phase_time,
            additional_info={
                "attack_phase_completed": True,
                "attacks_tried": [a.__class__.__name__ for a in self.attacks],
                "attack_results": [{"attack": type(a).__name__, "success": a.success} for a in attack_results]
            }
        )
    
    def verify_batch_with_attacks(self, network: torch.nn.Module, input_samples: torch.Tensor,
                                 config: VerificationConfig) -> List[VerificationResult]:
        """
        Verify multiple inputs using attack-guided approach
        
        Args:
            network: Neural network
            input_samples: Batch of input samples
            config: Verification configuration
            
        Returns:
            List of VerificationResult objects
        """
        results = []
        
        print(f"🔄 Batch verification: {input_samples.shape[0]} samples")
        
        for i in range(input_samples.shape[0]):
            print(f"\n📋 Sample {i+1}/{input_samples.shape[0]}")
            result = self.verify_with_attacks(network, input_samples[i:i+1], config)
            results.append(result)
            
            # Early statistics
            verified_count = sum(1 for r in results if r.verified)
            falsified_count = sum(1 for r in results if r.status == VerificationStatus.FALSIFIED)
            print(f"   Progress: {verified_count} verified, {falsified_count} falsified")
        
        return results
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get capabilities of the attack-guided engine"""
        capabilities = self.verifier.get_capabilities()
        capabilities.update({
            'verification_strategy': 'attack-guided',
            'attack_methods': [a.__class__.__name__ for a in self.attacks],
            'fast_falsification': True,
            'counterexample_generation': True,
            'adaptive_timeout': True,
            'batch_support': True
        })
        return capabilities
    
    def configure_attacks(self, attack_names: List[str], attack_timeout: Optional[float] = None):
        """
        Configure which attacks to use
        
        Args:
            attack_names: List of attack names to use
            attack_timeout: New attack timeout (optional)
        """
        from ..attacks import create_attack
        
        if attack_timeout is not None:
            self.attack_timeout = attack_timeout
        
        self.attacks = []
        for name in attack_names:
            try:
                attack = create_attack(name, self.device)
                self.attacks.append(attack)
                print(f"Added attack: {attack.__class__.__name__}")
            except Exception as e:
                print(f"Failed to add attack {name}: {e}")
        
        if not self.attacks:
            print("Warning: No attacks configured, falling back to FGSM")
            self.attacks = [FGSMAttack(self.device)]

# Factory functions for easy instantiation
def create_attack_guided_engine(device: str = 'cpu', 
                               attack_timeout: float = 10.0) -> AttackGuidedEngine:
    """Create attack-guided verification engine"""
    return AttackGuidedEngine(device=device, attack_timeout=attack_timeout)

def create_verification_engine_v2(use_attacks: bool = True, 
                                 device: str = 'cpu',
                                 attack_timeout: float = 10.0):
    """
    Enhanced factory function for verification engines
    
    Args:
        use_attacks: Whether to use attack-guided verification
        device: Device to run on
        attack_timeout: Timeout for attack phase
        
    Returns:
        Verification engine instance
    """
    if use_attacks:
        return create_attack_guided_engine(device, attack_timeout)
    else:
        from .alpha_beta_crown import AlphaBetaCrownEngine
        return AlphaBetaCrownEngine(device)