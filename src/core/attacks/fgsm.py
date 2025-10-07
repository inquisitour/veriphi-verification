# src/core/attacks/fgsm.py
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import (
    AdversarialAttack,
    AttackConfig,
    AttackResult,
    AttackStatus,
)

# -----------------------------
# helpers
# -----------------------------
def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
    return x if x.dim() > 1 else x.unsqueeze(0)


def _project_linf(x: torch.Tensor, x0: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.clamp(x, min=(x0 - eps), max=(x0 + eps))


def _project_l2(x: torch.Tensor, x0: torch.Tensor, eps: float) -> torch.Tensor:
    # project x into L2 ball centered at x0 with radius eps
    delta = x - x0
    flat = delta.view(delta.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    scale = (eps / norms).clamp(max=1.0)
    proj = (flat * scale).view_as(delta)
    return x0 + proj


def _softmax_confidence(logits: torch.Tensor, cls: int) -> float:
    probs = F.softmax(logits, dim=1)
    return float(probs[0, cls].item())


# =========================================================
# FGSM (single-step)
# =========================================================
class FGSMAttack(AdversarialAttack):
    def __init__(self, device: str | None = None) -> None:
        super().__init__(device=device)
        print(f"Initialized FGSM attack on device: {self.device}")

    # what tests expect
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "FGSMAttack",
            "iterative": False,
            "norms": ["inf", "2"],
        }

    def attack(
        self,
        model: nn.Module,
        input_sample: torch.Tensor,
        config: AttackConfig,
    ) -> AttackResult:
        model = model.to(self.device).eval()
        x_orig = _ensure_batch(input_sample).to(self.device).detach()
        eps = float(config.epsilon)
        norm = str(config.norm)

        loss_fn = nn.CrossEntropyLoss()
        start = time.time()

        # original prediction + confidence
        with torch.no_grad():
            logits0 = model(x_orig)
            orig_pred = int(logits0.argmax(dim=1).item())
            orig_conf = _softmax_confidence(logits0, orig_pred)

        print(f"FGSM Attack - Original: class {orig_pred} (conf: {orig_conf:.3f})")
        if config.targeted:
            print(f"Targeted attack: {orig_pred} -> {config.target_class}")
        else:
            print(f"Untargeted attack: trying to fool class {orig_pred}")

        # compute gradient wrt input
        x = x_orig.clone().requires_grad_(True)
        logits = model(x)

        if config.targeted:
            target = torch.tensor([int(config.target_class)], device=self.device)
            loss = loss_fn(logits, target)
            grad = torch.autograd.grad(loss, x)[0]
            grad_dir = -grad  # targeted: minimize loss toward target
        else:
            target = torch.tensor([orig_pred], device=self.device)
            loss = loss_fn(logits, target)
            grad = torch.autograd.grad(loss, x)[0]
            grad_dir = grad  # untargeted: maximize loss for original class

        # single-step update
        if norm == "inf":
            x_adv = x_orig + eps * torch.sign(grad_dir)
        elif norm == "2":
            g = grad_dir.view(grad_dir.size(0), -1)
            g_norm = g.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            step = (g / g_norm).view_as(grad_dir) * eps
            x_adv = x_orig + step
        else:
            raise ValueError(f"Unsupported norm: {norm}")

        # clip to valid range
        x_adv = x_adv.detach().clamp(config.clip_min, config.clip_max)

        with torch.no_grad():
            logits_adv = model(x_adv)
            adv_pred = int(logits_adv.argmax(dim=1).item())
            adv_conf = _softmax_confidence(logits_adv, adv_pred)

        success = (adv_pred == int(config.target_class)) if config.targeted else (adv_pred != orig_pred)
        status = AttackStatus.SUCCESS if success else AttackStatus.FAILED
        attack_time = time.time() - start

        print(f"Attack result: {'SUCCESS' if success else 'FAILED'}")

        # perturbation norm report
        pert_norm: Optional[float] = None
        try:
            delta = (x_adv - x_orig).view(x_orig.size(0), -1)
            if norm == "inf":
                pert_norm = float(delta.abs().max().item())
            elif norm == "2":
                pert_norm = float(delta.norm(p=2, dim=1).item())
        except Exception:
            pass

        additional_info = {
            "name": "FGSMAttack",
            "epsilon": eps,
            "norm": norm,
            "targeted": bool(config.targeted),
            "target_class": int(config.target_class) if config.target_class is not None else None,
            "loss_function": "cross_entropy",
            "confidence_drop": (orig_conf - adv_conf),
        }

        return AttackResult(
            success=success,
            status=status,
            adversarial_example=x_adv.detach().cpu(),
            original_prediction=orig_pred,
            adversarial_prediction=adv_pred,
            perturbation_norm=pert_norm,
            attack_time=attack_time,
            iterations_used=1,
            additional_info=additional_info,
        )


# =========================================================
# I-FGSM (multi-step)
# =========================================================
class IterativeFGSM(AdversarialAttack):
    def __init__(self, device: str | None = None) -> None:
        super().__init__(device=device)
        print(f"Initialized I-FGSM attack on device: {self.device}")

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "IterativeFGSM",
            "iterative": True,
            "norms": ["inf", "2"],
        }

    def attack(
        self,
        model: nn.Module,
        input_sample: torch.Tensor,
        config: AttackConfig,
    ) -> AttackResult:
        model = model.to(self.device).eval()
        x_orig = _ensure_batch(input_sample).to(self.device).detach()

        max_iters = int(max(1, config.max_iterations or 10))
        step = float(config.step_size) if config.step_size is not None else float(config.epsilon) / max_iters
        eps = float(config.epsilon)
        norm = str(config.norm)

        x_adv = x_orig.clone()
        if getattr(config, "random_start", False):
            # L∞ random start; for L2, this is a reasonable simple variant
            noise = torch.empty_like(x_orig).uniform_(-eps, eps)
            x_adv = torch.clamp(x_orig + noise, config.clip_min, config.clip_max)

        loss_fn = nn.CrossEntropyLoss()
        start = time.time()

        with torch.no_grad():
            logits0 = model(x_orig)
            orig_pred = int(logits0.argmax(dim=1).item())
            orig_conf = _softmax_confidence(logits0, orig_pred)

        print(f"FGSM Attack - Original: class {orig_pred} (conf: {orig_conf:.3f})")
        if config.targeted:
            print("Targeted attack: {} -> {}".format(orig_pred, config.target_class))
        else:
            print(f"Untargeted attack: trying to fool class {orig_pred}")

        success = False
        adv_pred = orig_pred
        iterations_used = 0

        for it in range(max_iters):
            x_adv.requires_grad_(True)
            logits = model(x_adv)

            if config.targeted:
                target = torch.tensor([int(config.target_class)], device=self.device)
                loss = loss_fn(logits, target)
                grad = torch.autograd.grad(loss, x_adv)[0]
                grad_dir = -grad  # targeted: minimize loss toward target
            else:
                target = torch.tensor([orig_pred], device=self.device)
                loss = loss_fn(logits, target)
                grad = torch.autograd.grad(loss, x_adv)[0]
                grad_dir = grad  # untargeted: maximize loss for original class

            if norm == "inf":
                x_adv = x_adv + step * torch.sign(grad_dir)
                x_adv = _project_linf(x_adv, x_orig, eps)
            elif norm == "2":
                g = grad_dir.view(grad_dir.size(0), -1)
                g_norm = g.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
                step_vec = (g / g_norm).view_as(grad_dir) * step
                if config.targeted:
                    step_vec = -step_vec
                x_adv = x_adv + step_vec
                x_adv = _project_l2(x_adv, x_orig, eps)
            else:
                raise ValueError(f"Unsupported norm: {norm}")

            x_adv = x_adv.detach().clamp(config.clip_min, config.clip_max)

            with torch.no_grad():
                logits_adv = model(x_adv)
                adv_pred = int(logits_adv.argmax(dim=1).item())

            success = (adv_pred == int(config.target_class)) if config.targeted else (adv_pred != orig_pred)
            iterations_used = it + 1

            # only break if early_stopping is enabled and we already succeeded
            if getattr(config, "early_stopping", True) and success:
                break
            # otherwise keep going to max_iters

        attack_time = time.time() - start

        # when early_stopping is False, report all iterations used
        if not getattr(config, "early_stopping", True):
            iterations_used = max_iters

        status = AttackStatus.SUCCESS if success else AttackStatus.FAILED
        print(f"Attack result: {'SUCCESS' if success else 'FAILED'}")

        # perturbation norm
        pert_norm: Optional[float] = None
        try:
            delta = (x_adv - x_orig).view(x_orig.size(0), -1)
            if norm == "inf":
                pert_norm = float(delta.abs().max().item())
            elif norm == "2":
                pert_norm = float(delta.norm(p=2, dim=1).item())
        except Exception:
            pass

        # confidence drop (only for untargeted we compare original class)
        with torch.no_grad():
            logits_final = model(x_adv)
            if config.targeted and config.target_class is not None:
                conf_ref = _softmax_confidence(logits_final, int(config.target_class))
            else:
                conf_ref = _softmax_confidence(logits_final, adv_pred)
        confidence_drop = None
        try:
            confidence_drop = (orig_conf - conf_ref)
        except Exception:
            pass

        additional_info = {
            "name": "IterativeFGSM",
            "epsilon": eps,
            "norm": norm,
            "targeted": bool(config.targeted),
            "target_class": int(config.target_class) if config.target_class is not None else None,
            "loss_function": "cross_entropy",
            "step_size": step,
            "confidence_drop": confidence_drop,
        }

        return AttackResult(
            success=success,
            status=status,
            adversarial_example=x_adv.detach().cpu(),
            original_prediction=orig_pred,
            adversarial_prediction=adv_pred,
            perturbation_norm=pert_norm,
            attack_time=attack_time,
            iterations_used=iterations_used,
            additional_info=additional_info,
        )
