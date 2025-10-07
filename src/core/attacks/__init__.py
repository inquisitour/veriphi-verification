# src/core/attacks/__init__.py
"""
Adversarial Attacks Module
"""

import os
from .base import (
    AdversarialAttack,
    AttackResult, 
    AttackConfig, 
    AttackStatus,
    TargetedAttackMixin,
    UntargetedAttackMixin,
    AttackMetrics
)
from .fgsm import (
    FGSMAttack,
    IterativeFGSM
)

__all__ = [
    'AdversarialAttack',
    'AttackResult', 
    'AttackConfig', 
    'AttackStatus',
    'TargetedAttackMixin',
    'UntargetedAttackMixin',
    'AttackMetrics',
    'FGSMAttack',
    'IterativeFGSM',
    'create_attack',
    'list_available_attacks',
]

__version__ = '0.1.1'

# registry
ATTACK_REGISTRY = {
    'fgsm': FGSMAttack,
    'i-fgsm': IterativeFGSM,
    'ifgsm': IterativeFGSM,
    'bim': IterativeFGSM,
}

# default device
DEFAULT_DEVICE = os.environ.get("VERIPHI_DEVICE", "cpu")

def create_attack(attack_name: str, device: str | None = None) -> AdversarialAttack:
    """
    Factory function to create attacks by name.
    Defaults to VERIPHI_DEVICE if device is None.
    """
    if attack_name.lower() not in ATTACK_REGISTRY:
        available = list(ATTACK_REGISTRY.keys())
        raise ValueError(f"Unknown attack: {attack_name}. Available: {available}")

    device = device or DEFAULT_DEVICE
    attack_class = ATTACK_REGISTRY[attack_name.lower()]
    return attack_class(device=device)

def list_available_attacks() -> list[str]:
    """Return list of available attack names"""
    return list(ATTACK_REGISTRY.keys())
