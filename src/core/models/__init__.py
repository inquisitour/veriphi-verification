# src/core/models/__init__.py
"""
Neural Network Models Module

This module provides test models and utilities for neural network verification.
"""

from .test_models import (
    SimpleLinearNet,
    SimpleConvNet,
    TinyNet,
    DeepLinearNet,
    create_test_model,
    initialize_model_weights,
    load_pretrained_model,
    create_sample_input,
    create_model_from_config,
    MODEL_CONFIGS
)

__all__ = [
    # Model classes
    'SimpleLinearNet',
    'SimpleConvNet', 
    'TinyNet',
    'DeepLinearNet',
    
    # Factory functions
    'create_test_model',
    'create_model_from_config',
    'create_sample_input',
    
    # Utilities
    'initialize_model_weights',
    'load_pretrained_model',
    
    # Configurations
    'MODEL_CONFIGS'
]
