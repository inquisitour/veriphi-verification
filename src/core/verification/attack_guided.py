# src/core/verification/attack_guided.py
from __future__ import annotations

import contextlib
import time
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

# --- Optional CPU memory path ---
try:
    import psutil  # noqa: F401
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

from .base import (
    VerificationEngine,
    VerificationConfig,
    VerificationResult,
    VerificationStatus,
)
from .alpha_beta_crown import AlphaBetaCrownEngine
from ..attacks import (
    AttackConfig,
    AttackResult,
    create_attack,
    list_available_attacks,
)


def _cpu_mem_mb() -> Optional[float]:
    """Return current process RSS in MB (CPU path)."""
    if psutil is None:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss / 1024.0 / 1024.0
    except Exception:
        return None


@dataclass
class _AttackRunRecord:
    name: str
    success: bool
    status: str
    time_s: float


class AttackGuidedEngine(VerificationEngine):
    """
    Hybrid verifier:
      1) Try fast adversarial attacks for quick falsification
      2) If no counterexample, call the formal verifier (α,β-CROWN)
    """

    def __init__(self, device: Optional[str] = None, attack_timeout: float = 10.0) -> None:
        # Resolve device from arg or environment variable
        if device is None:
            device = os.environ.get("VERIPHI_DEVICE", "cpu")
        # store torch.device for tensor operations
        self.device = torch.device(device)
        self.attack_timeout = float(attack_timeout)

        # Formal engine (pass device as string to factories that expect it)
        self.formal_engine = AlphaBetaCrownEngine(device=str(self.device))

        # Instantiate default attacks (device passed as string)
        self.attacks = [
            create_attack("fgsm", device=str(self.device)),
            create_attack("i-fgsm", device=str(self.device)),
        ]

        print(f"✓ Attack-guided verification engine initialized on {self.device}")

    # ------------------------------------------------------------------ #
    # Public API expected by tests
    # ------------------------------------------------------------------ #
    def get_capabilities(self) -> Dict[str, Any]:
        caps: Dict[str, Any] = {}
        try:
            caps.update(self.formal_engine.get_capabilities())
        except Exception:
            pass

        caps.update(
            {
                "verification_strategy": "attack-guided",
                "attack_timeout": self.attack_timeout,
                "attack_methods": [a.__class__.__name__ for a in self.attacks],
                "fast_falsification": True,
                "norms": ["inf", "2"],
                "bound_methods": ["IBP", "CROWN", "alpha-CROWN"],
                "device": str(self.device),
            }
        )
        return caps

    def verify(self, model: torch.nn.Module, input_sample: torch.Tensor, config: VerificationConfig) -> VerificationResult:
        return self.verify_with_attacks(model, input_sample, config)

    # ------------------------------------------------------------------ #
    # Core flow
    # ------------------------------------------------------------------ #
    def verify_with_attacks(
        self,
        model: torch.nn.Module,
        input_sample: torch.Tensor,
        config: VerificationConfig,
    ) -> VerificationResult:
        print("🚀 Starting attack-guided verification")
        print(f"   Property: ε={config.epsilon}, norm=L{config.norm if config.norm != 'inf' else 'inf'}")
        print(f"   Attack timeout: {self.attack_timeout:.1f}s")
        print(f"   Verification timeout: {config.timeout}s")
        print(f"   🗡️ Phase 1: Attack phase (timeout: {self.attack_timeout:.1f}s)")

        attacks_tried: List[str] = []
        attack_success: Optional[AttackResult] = None

        atk_cfg = AttackConfig(
            epsilon=float(config.epsilon),
            norm=str(config.norm),
            targeted=False,
            max_iterations=5,
            early_stopping=True,
        )

        # Move model and input to correct device up front
        model = model.to(self.device).eval()
        input_sample = input_sample.to(self.device)

        attack_t0 = time.time()
        for attack in self.attacks:
            name = attack.__class__.__name__
            attacks_tried.append(name)
            print(f"      Trying {name}...")
            try:
                res = attack.attack(model, input_sample, atk_cfg)
            except Exception as e:
                print(f"      ○ {name} raised an exception during attack: {e}")
                res = AttackResult(success=False, perturbation=None, additional_info={"error": str(e)})

            # Normalize AttackResult interface (duck-typed)
            if isinstance(res, AttackResult):
                if getattr(res, "success", False):
                    print("   ⚠️  Counterexample found by attack; skipping formal verification")
                    attack_success = res
                    break
                else:
                    print(f"      ○ {name} failed to find counterexample")
            else:
                # If attack returned unexpected type, treat as failure
                print(f"      ○ {name} returned unexpected result type; treating as failure")

        attack_time = time.time() - attack_t0
        attack_phase_result = "counterexample_found" if attack_success else "no_counterexample_found"
        print(f"   ○ Attack phase completed ({attack_time:.3f}s) - {'Counterexample found' if attack_success else 'No counterexamples found'}")

        # --- If attack succeeded ---
        if attack_success is not None:
            return self._make_falsified_result_from_attack(
                attack_success,
                attacks_tried,
                attack_time,
            )

        # --- Proceed to formal verification ---
        print("   ⚡ Attacks completed, proceeding with formal verification...")

        use_cuda = (self.device.type == "cuda" and torch.cuda.is_available())
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.time()
        with contextlib.ExitStack() as stack:
            if use_cuda:
                # Use autocast for memory/speed improvements if supported
                stack.enter_context(torch.cuda.amp.autocast(dtype=torch.float16))
            formal_result = self.formal_engine.verify(model, input_sample, config)

        if use_cuda:
            torch.cuda.synchronize()
        dt = time.time() - t0

        if use_cuda:
            peak_bytes = torch.cuda.max_memory_allocated()
            mem_mb = float(peak_bytes) / (1024.0 * 1024.0)
        else:
            mem_mb = _cpu_mem_mb()

        # attach timing and memory info
        formal_result.verification_time = dt
        formal_result.memory_usage = mem_mb
        info = dict(formal_result.additional_info or {})
        info.update(
            {
                "attacks_tried": attacks_tried,
                "attack_phase_time": attack_time,
                "attack_phase_result": attack_phase_result,
                "attack_phase_completed": True,
                "phase_completed": True,
                "memory_usage_mb": mem_mb,
                "method": "attack-guided",
                "verification_method": "attack-guided",
            }
        )
        formal_result.additional_info = info
        return formal_result

    # ------------------------------------------------------------------ #
    # Batch API (added)
    # ------------------------------------------------------------------ #
    def verify_batch_with_attacks(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        config: VerificationConfig,
    ) -> List[VerificationResult]:
        """Run attack-guided verification on a batch of inputs."""
        results: List[VerificationResult] = []
        # Move model to device once
        model = model.to(self.device).eval()
        for i in range(inputs.size(0)):
            x = inputs[i : i + 1].to(self.device)
            res = self.verify_with_attacks(model, x, config)
            results.append(res)
        return results

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _make_falsified_result_from_attack(
        self,
        attack_res: AttackResult,
        attacks_tried: List[str],
        attack_phase_time: float,
    ) -> VerificationResult:
        # Extract attack metadata safely
        falsified_by = "attack"
        attack_name = "unknown"
        extra_info = {}
        if isinstance(attack_res, AttackResult):
            falsified_by = "attack"
            extra_info = attack_res.additional_info or {}
            attack_name = extra_info.get("name", getattr(attack_res, "name", "unknown"))

        result = VerificationResult(
            verified=False,
            status=VerificationStatus.FALSIFIED,
            bounds=None,
            verification_time=attack_phase_time,
            additional_info={
                "attacks_tried": attacks_tried,
                "attack_phase_time": attack_phase_time,
                "attack_phase_result": "counterexample_found",
                "attack_phase_completed": True,
                "phase_completed": True,
                "method": "attack-guided",
                "verification_method": "attack-guided",
                "falsified_by": falsified_by,
                "attack_name": attack_name,
                "attack_additional_info": extra_info,
            },
        )
        result.memory_usage = _cpu_mem_mb()
        return result


# Factory
def create_attack_guided_engine(device: Optional[str] = None, attack_timeout: float = 10.0) -> AttackGuidedEngine:
    """Factory that respects VERIPHI_DEVICE if device is not provided."""
    if device is None:
        device = os.environ.get("VERIPHI_DEVICE", "cpu")
    print("✓ Attack-guided verification engine factory creating engine")
    return AttackGuidedEngine(device=device, attack_timeout=attack_timeout)
