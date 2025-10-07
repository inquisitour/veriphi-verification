# src/core/verification/__init__.py
"""
Neural Network Verification Module

This module provides core verification functionality using α,β-CROWN algorithm
via the auto-LiRPA library.
"""
import os
DEFAULT_DEVICE = os.environ.get("VERIPHI_DEVICE", "cpu")

from .base import (
    VerificationEngine, 
    VerificationResult, 
    VerificationConfig, 
    VerificationStatus,
    PropertySpecification,
    ModelInfo
)

from .alpha_beta_crown import (
    AlphaBetaCrownEngine,
    GPUOptimizedEngine,
    create_verification_engine
)

__all__ = [
    # Base classes and data structures
    'VerificationEngine', 
    'VerificationResult', 
    'VerificationConfig', 
    'VerificationStatus',
    'PropertySpecification',
    'ModelInfo',
    
    # Engine implementations
    'AlphaBetaCrownEngine',
    'GPUOptimizedEngine',
    'create_verification_engine'
]

# Version info
__version__ = '0.1.0'

# Default engine factory
def get_default_engine(device: str | None = None) -> VerificationEngine:
    """Get the default verification engine"""
    device = device or DEFAULT_DEVICE
    return create_verification_engine(device=device, optimized=False)
