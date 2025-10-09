# src/core/verification/alpha_beta_crown.py
from __future__ import annotations

import os
import time
import tracemalloc
import contextlib
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np

# auto-LiRPA imports
from auto_LiRPA.bound_general import BoundedModule
from auto_LiRPA.bounded_tensor import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from .base import VerificationEngine, VerificationResult, VerificationConfig, VerificationStatus


class AlphaBetaCrownEngine(VerificationEngine):
    """α,β-CROWN verification implementation using auto-LiRPA"""

    def __init__(self, device: Optional[str] = None):
        """Initialize α,β-CROWN verifier."""
        # resolve from environment if not given
        if device is None:
            device = os.environ.get("VERIPHI_DEVICE", "cpu")
        self.device = torch.device(device)
        print(f"Initializing α,β-CROWN verifier on device: {self.device}")

        # Default optimization settings for both old and new auto-LiRPA APIs
        self.default_bound_opts = {
            # Newer API (>=0.6.1)
            "bound_opts": {
                "optimizer": "adam",
                "lr_alpha": 0.1,
                "iteration": 20,
            },
            # Legacy API (<=0.6.0)
            "optimize_bound_args": {
                "optimizer": "adam",
                "lr": 0.1,
                "iteration": 20,
            },
        }

    # ------------------------------------------------------------------
    # main verify routine
    # ------------------------------------------------------------------
    def verify(
        self, network: nn.Module, input_sample: torch.Tensor, config: VerificationConfig
    ) -> VerificationResult:
        """Verify robustness using α,β-CROWN."""
        start_time = time.time()
        tracemalloc.start()

        try:
            # preprocess + move to device
            input_sample = self._preprocess_input(input_sample)
            network = network.to(self.device).eval()
            input_sample = input_sample.to(self.device)

            # original prediction
            with torch.no_grad():
                output = network(input_sample)
                predicted_class = torch.argmax(output, dim=1)

            print(f"Original prediction: class {predicted_class.item()}")
            print(
                f"Verifying robustness with ε={config.epsilon}, norm=L{config.norm}, bound={config.bound_method}"
            )

            # build bounded model
            bounded_model = BoundedModule(network, input_sample)
            perturbation = self._create_perturbation(config)
            bounded_input = BoundedTensor(input_sample, perturbation)

            # compute bounds
            lb, ub = self._compute_bounds(bounded_model, bounded_input, config)

            # check robustness
            verified = self._check_robustness(lb, ub, predicted_class)

            verification_time = time.time() - start_time
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            status = (
                VerificationStatus.VERIFIED
                if verified
                else VerificationStatus.FALSIFIED
            )

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
                    "bounds_gap": self._compute_bounds_gap(lb, ub, predicted_class),
                    "device": str(self.device),
                },
            )

        except Exception as e:
            verification_time = time.time() - start_time
            with contextlib.suppress(Exception):
                tracemalloc.stop()
            print(f"Verification failed with error: {e}")
            _, peak_memory = tracemalloc.get_traced_memory()
            return VerificationResult(
                verified=False,
                status=VerificationStatus.ERROR,
                verification_time=verification_time,
                memory_usage=peak_memory / 1024 / 1024,  # MB
                additional_info={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "device": str(self.device),
                    "perturbation_norm": getattr(config, "norm", None),
                },
            )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            x = x.unsqueeze(0)
        if x.min() < 0 or x.max() > 1:
            if x.max() > 1:
                x = x / 255.0
            else:
                x = (x + 1) / 2.0
        return x

    def _create_perturbation(self, config: VerificationConfig) -> PerturbationLpNorm:
        if config.norm == "inf":
            return PerturbationLpNorm(norm=np.inf, eps=config.epsilon)
        elif config.norm == "2":
            return PerturbationLpNorm(norm=2, eps=config.epsilon)
        else:
            raise ValueError(f"Unsupported norm: {config.norm}")

    def _compute_bounds(
    self, bounded_model: BoundedModule, bounded_input: BoundedTensor, config: VerificationConfig
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        method_map = {
            "IBP": "IBP",
            "CROWN": "backward",
            "alpha-CROWN": "alpha-crown",
            "beta-CROWN": "beta-crown",
        }
        method = method_map.get(config.bound_method, "alpha-crown")
        print(f"Computing bounds using method: {method}")

        # Universal fallback for different auto-LiRPA API variants
        if method in ["alpha-crown", "beta-crown"]:
            try:
                # 1️⃣ Newest API (>=0.7.x)
                lb, ub = bounded_model.compute_bounds(
                    x=(bounded_input,),
                    method=method,
                    bound_upper=True,
                    **{"bound_opts": self.default_bound_opts["bound_opts"]},
                )
            except TypeError as e1:
                try:
                    # 2️⃣ Mid API (≈0.6.x)
                    print("⚙️ Falling back to optimize_bound_args API...")
                    lb, ub = bounded_model.compute_bounds(
                        x=(bounded_input,),
                        method=method,
                        bound_upper=True,
                        **{"optimize_bound_args": self.default_bound_opts["optimize_bound_args"]},
                    )
                except TypeError as e2:
                    try:
                        # 3️⃣ Legacy API (≤0.5.x): use simplified backward pass
                        print("⚙️ Falling back to legacy non-optimized CROWN computation...")
                        lb, ub = bounded_model.compute_bounds(
                            x=(bounded_input,),
                            method="backward",  # old α/β-CROWN fallback
                            bound_upper=True,
                        )
                    except Exception as e3:
                        raise RuntimeError(
                            f"All α/β-CROWN bound computation attempts failed:\n"
                            f"  bound_opts → {e1}\n"
                            f"  optimize_bound_args → {e2}\n"
                            f"  backward → {e3}"
                        )
        else:
            lb, ub = bounded_model.compute_bounds(
                x=(bounded_input,),
                method=method,
                bound_upper=True,
            )

    def _check_robustness(
        self, lb: torch.Tensor, ub: torch.Tensor, predicted_class: torch.Tensor
    ) -> bool:
        batch_size = lb.shape[0]
        for i in range(batch_size):
            pc = predicted_class[i].item()
            lb_pred = lb[i, pc]
            for j in range(lb.shape[1]):
                if j != pc and lb_pred <= ub[i, j]:
                    print(
                        f"Robustness violated: class {pc} LB({lb_pred:.6f}) <= class {j} UB({ub[i,j]:.6f})"
                    )
                    return False
        print("Robustness verified: predicted class maintains highest confidence")
        return True

    def _compute_bounds_gap(
        self, lb: torch.Tensor, ub: torch.Tensor, predicted_class: torch.Tensor
    ) -> float:
        pc = predicted_class[0].item()
        lb_pred = lb[0, pc].item()
        others = [ub[0, j].item() for j in range(lb.shape[1]) if j != pc]
        max_other = max(others) if others else 0.0
        return lb_pred - max_other

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "layers": [
                "Linear",
                "Conv2d",
                "ReLU",
                "MaxPool2d",
                "AvgPool2d",
                "BatchNorm2d",
                "Flatten",
            ],
            "activations": ["ReLU", "Sigmoid", "Tanh"],
            "norms": ["inf", "2"],
            "specifications": ["robustness", "reachability"],
            "bound_methods": ["IBP", "CROWN", "alpha-CROWN", "beta-CROWN"],
            "max_neurons": 1_000_000,
            "gpu_accelerated": self.device.type == "cuda",
            "supports_batch": False,
            "framework": "auto-LiRPA",
            "algorithm": "α,β-CROWN",
            "device": str(self.device),
        }


class GPUOptimizedEngine(AlphaBetaCrownEngine):
    """Enhanced verifier with memory optimizations for large models."""

    def __init__(self, device: Optional[str] = None):
        super().__init__(device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            print("GPU optimizations enabled for α,β-CROWN")

    def verify_large_model(
        self, network: nn.Module, input_sample: torch.Tensor, config: VerificationConfig
    ) -> VerificationResult:
        if hasattr(network, "gradient_checkpointing_enable"):
            network.gradient_checkpointing_enable()
        if self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                return self.verify(network, input_sample, config)
        return self.verify(network, input_sample, config)


def create_verification_engine(
    device: Optional[str] = None, optimized: bool = False
) -> VerificationEngine:
    """Factory function that respects VERIPHI_DEVICE environment variable."""
    if device is None:
        device = os.environ.get("VERIPHI_DEVICE", "cpu")
    return GPUOptimizedEngine(device) if optimized else AlphaBetaCrownEngine(device)
