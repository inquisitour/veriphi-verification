# src/core/attacks/__init__.py
"""
Adversarial Attacks Module

This module provides implementations of various adversarial attack methods
for neural network robustness evaluation.
"""

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
    # Base classes and data structures
    'AdversarialAttack',
    'AttackResult', 
    'AttackConfig', 
    'AttackStatus',
    'TargetedAttackMixin',
    'UntargetedAttackMixin',
    'AttackMetrics',
    
    # Attack implementations
    'FGSMAttack',
    'IterativeFGSM'
]

# Version info
__version__ = '0.1.0'

# Attack registry for easy instantiation
ATTACK_REGISTRY = {
    'fgsm': FGSMAttack,
    'i-fgsm': IterativeFGSM,
    'ifgsm': IterativeFGSM,  # Alternative name
    'bim': IterativeFGSM,    # Basic Iterative Method
}

def create_attack(attack_name: str, device: str = 'cpu') -> AdversarialAttack:
    """
    Factory function to create attacks by name
    
    Args:
        attack_name: Name of the attack ('fgsm', 'i-fgsm')
        device: Device to run on ('cpu', 'cuda')
        
    Returns:
        AdversarialAttack instance
    """
    if attack_name.lower() not in ATTACK_REGISTRY:
        available = list(ATTACK_REGISTRY.keys())
        raise ValueError(f"Unknown attack: {attack_name}. Available: {available}")
    
    attack_class = ATTACK_REGISTRY[attack_name.lower()]
    return attack_class(device=device)

def list_available_attacks() -> list:
    """Return list of available attack names"""
    return list(ATTACK_REGISTRY.keys())