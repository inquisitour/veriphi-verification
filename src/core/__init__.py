from __future__ import annotations

from typing import List, Optional, Any
import torch

# Verification
from .verification.base import (
    VerificationEngine,
    VerificationResult,
    VerificationConfig,
    VerificationStatus,
    PropertySpecification,
    ModelInfo,
)
from .verification.alpha_beta_crown import AlphaBetaCrownEngine
from .verification.attack_guided import AttackGuidedEngine, create_attack_guided_engine

# Attacks
from .attacks.base import (
    AdversarialAttack,
    AttackResult,
    AttackConfig,
    AttackStatus,
    AttackMetrics,
)
from .attacks import create_attack, list_available_attacks

# Models
from .models.test_models import (
    create_test_model,
    create_sample_input,
    create_model_from_config,
    MODEL_CONFIGS,
)


class _StatusCompat:
    def __init__(self, status: Any):
        if hasattr(status, "value"):
            self.value = status.value
            self._enum = status
        else:
            self.value = str(status)
            self._enum = None

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.value == other
        if hasattr(other, "value"):
            return self.value == other.value
        return False

    def __repr__(self) -> str:
        return f"StatusCompat({self.value})"


class VeriphiCore:
    def __init__(self, use_attacks: bool = True, device: str = "cpu", attack_timeout: float = 10.0):
        self.use_attacks = use_attacks
        self.device = device
        self.attack_timeout = attack_timeout
        if use_attacks:
            self.engine = create_attack_guided_engine(device, attack_timeout)
            print("✓ Attack-guided verification engine initialized")
        else:
            self.engine = AlphaBetaCrownEngine(device)
            print("✓ Formal verification engine initialized")

    def verify_robustness(self, model: torch.nn.Module, input_sample: torch.Tensor,
                          epsilon: float = 0.1, norm: str = "inf", timeout: int = 300,
                          bound_method: str = "CROWN") -> VerificationResult:
        config = VerificationConfig(
            epsilon=epsilon, norm=norm, timeout=timeout,
            method="alpha-beta-crown", bound_method=bound_method
        )
        if hasattr(self.engine, "verify_with_attacks"):
            return self.engine.verify_with_attacks(model, input_sample, config)
        return self.engine.verify(model, input_sample, config)

    def verify_batch(self, model: torch.nn.Module, input_samples: torch.Tensor,
                     epsilon: float = 0.1, norm: str = "inf", timeout: int = 300) -> List[VerificationResult]:
        config = VerificationConfig(
            epsilon=epsilon, norm=norm,
            timeout=timeout // max(1, int(input_samples.shape[0])),
            method="alpha-beta-crown",
        )
        if hasattr(self.engine, "verify_batch_with_attacks"):
            return self.engine.verify_batch_with_attacks(model, input_samples, config)
        return self.engine.verify_batch(model, input_samples, config)

    def attack_model(self, model: torch.nn.Module, input_sample: torch.Tensor,
                     attack_name: str = "fgsm", epsilon: float = 0.1, norm: str = "inf",
                     targeted: bool = False, target_class: Optional[int] = None) -> AttackResult:
        attack = create_attack(attack_name, self.device)
        config = AttackConfig(
            epsilon=epsilon, norm=norm,
            targeted=targeted, target_class=target_class,
            max_iterations=20,
        )
        result = attack.attack(model, input_sample, config)
        # Make status work with both test styles
        result.status = _StatusCompat(result.status)
        return result

    def evaluate_robustness(self, model: torch.nn.Module, test_inputs: torch.Tensor,
                            epsilons: List[float] = [0.01, 0.05, 0.1, 0.2], norm: str = "inf") -> dict:
        results = {}
        print(f"🔍 Evaluating robustness across {len(epsilons)} epsilon values")
        print(f"   Test samples: {test_inputs.shape[0]}")
        print(f"   Epsilons: {epsilons}")
        for eps in epsilons:
            print(f"\n📊 Testing ε = {eps}")
            verification_results = self.verify_batch(model, test_inputs, epsilon=eps, norm=norm, timeout=60)
            total_samples = len(verification_results)
            verified_count = sum(1 for r in verification_results if r.verified)
            falsified_count = sum(1 for r in verification_results if r.status == VerificationStatus.FALSIFIED)
            error_count = sum(1 for r in verification_results if r.status == VerificationStatus.ERROR)
            verification_rate = verified_count / total_samples
            falsification_rate = falsified_count / total_samples
            avg_time = sum(r.verification_time or 0 for r in verification_results) / total_samples
            results[eps] = {
                "verification_rate": verification_rate,
                "falsification_rate": falsification_rate,
                "error_rate": error_count / total_samples,
                "verified_count": verified_count,
                "falsified_count": falsified_count,
                "error_count": error_count,
                "total_samples": total_samples,
                "average_time": avg_time,
                "results": verification_results,
            }
            print(f"   Verified: {verified_count}/{total_samples} ({verification_rate:.1%})")
            print(f"   Falsified: {falsified_count}/{total_samples} ({falsification_rate:.1%})")
            print(f"   Average time: {avg_time:.3f}s")
        return results

    def get_capabilities(self) -> dict:
        caps = self.engine.get_capabilities()
        caps.update({
            "attack_support": self.use_attacks,
            "available_attacks": list_available_attacks(),
            "batch_verification": True,
            "robustness_evaluation": True,
        })
        return caps


def create_core_system(use_attacks: bool = True, device: str = "cpu") -> VeriphiCore:
    return VeriphiCore(use_attacks=use_attacks, device=device)


def quick_robustness_check(model: torch.nn.Module, input_sample: torch.Tensor, epsilon: float = 0.1) -> bool:
    core = create_core_system(use_attacks=True, device="cpu")
    result = core.verify_robustness(model, input_sample, epsilon=epsilon, timeout=30)
    return result.verified


__all__ = [
    "VeriphiCore",
    "create_core_system",
    "quick_robustness_check",
    "VerificationEngine",
    "VerificationResult",
    "VerificationConfig",
    "VerificationStatus",
    "AlphaBetaCrownEngine",
    "AttackGuidedEngine",
    "PropertySpecification",
    "ModelInfo",
    "AdversarialAttack",
    "AttackResult",
    "AttackConfig",
    "AttackStatus",
    "AttackMetrics",
    "create_attack",
    "list_available_attacks",
    "create_test_model",
    "create_sample_input",
    "create_model_from_config",
    "MODEL_CONFIGS",
    "create_attack_guided_engine",
]
