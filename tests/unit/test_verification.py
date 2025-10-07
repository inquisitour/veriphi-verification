# tests/unit/test_verification.py
"""
Unit tests for verification components.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
import time

DEFAULT_DEVICE = os.environ.get("VERIPHI_DEVICE", "cpu")

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.verification import (
    VerificationEngine, VerificationResult, VerificationConfig, VerificationStatus,
    AlphaBetaCrownEngine, create_verification_engine
)
from core.models import create_test_model, create_sample_input


class TestVerificationConfig:
    """Test verification configuration validation"""
    
    def test_valid_config(self):
        """Test valid configuration creation"""
        config = VerificationConfig(epsilon=0.1, norm="inf")
        assert config.epsilon == 0.1
        assert config.norm == "inf"
        assert config.timeout == 300  # default
    
    def test_invalid_epsilon(self):
        """Test invalid epsilon values"""
        with pytest.raises(ValueError):
            VerificationConfig(epsilon=-0.1)
        
        with pytest.raises(ValueError):
            VerificationConfig(epsilon=0.0)
    
    def test_invalid_norm(self):
        """Test invalid norm values"""
        with pytest.raises(ValueError):
            VerificationConfig(norm="invalid")
    
    def test_invalid_timeout(self):
        """Test invalid timeout values"""
        with pytest.raises(ValueError):
            VerificationConfig(timeout=-10)
    
    def test_invalid_bound_method(self):
        """Test invalid bound method"""
        with pytest.raises(ValueError):
            VerificationConfig(bound_method="invalid")


class TestVerificationResult:
    """Test verification result structure"""
    
    def test_valid_result(self):
        """Test valid result creation"""
        result = VerificationResult(
            verified=True,
            status=VerificationStatus.VERIFIED,
            verification_time=1.5
        )
        assert result.verified is True
        assert result.status == VerificationStatus.VERIFIED
        assert result.verification_time == 1.5
    
    def test_inconsistent_result_verified_true(self):
        """Test inconsistent result: verified=True but status=FALSIFIED"""
        with pytest.raises(ValueError):
            VerificationResult(
                verified=True,
                status=VerificationStatus.FALSIFIED
            )
    
    def test_inconsistent_result_verified_false(self):
        """Test inconsistent result: verified=False but status=VERIFIED"""
        with pytest.raises(ValueError):
            VerificationResult(
                verified=False,
                status=VerificationStatus.VERIFIED
            )


class TestAlphaBetaCrownEngine:
    """Test α,β-CROWN verification engine"""
    
    @pytest.fixture
    def engine(self):
        """Create verification engine for testing"""
        return AlphaBetaCrownEngine(device=DEFAULT_DEVICE)
    
    @pytest.fixture
    def simple_model(self):
        """Create simple test model"""
        return create_test_model("tiny")
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor"""
        return create_sample_input("tiny")
    
    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine is not None
        assert engine.device.type == DEFAULT_DEVICE
        
        capabilities = engine.get_capabilities()
        assert 'layers' in capabilities
        assert 'activations' in capabilities
        assert 'norms' in capabilities
        assert 'bound_methods' in capabilities
    
    def test_engine_capabilities(self, engine):
        """Test engine capabilities"""
        capabilities = engine.get_capabilities()
        
        # Check required capabilities
        assert 'ReLU' in capabilities['activations']
        assert 'Linear' in capabilities['layers']
        assert 'inf' in capabilities['norms']
        assert '2' in capabilities['norms']
        assert 'IBP' in capabilities['bound_methods']
        assert 'CROWN' in capabilities['bound_methods']
        assert 'alpha-CROWN' in capabilities['bound_methods']
    
    def test_verification_small_epsilon(self, engine, simple_model, sample_input):
        """Test verification with small epsilon (likely to verify)"""
        config = VerificationConfig(epsilon=0.001, norm="inf", timeout=30)
        
        result = engine.verify(simple_model, sample_input, config)
        
        # Check result structure
        assert isinstance(result, VerificationResult)
        assert result.status.value in [s.value for s in VerificationStatus]
        assert result.verification_time is not None
        assert result.verification_time > 0
        assert result.memory_usage is not None
        
        # Small epsilon should not error
        assert result.status != VerificationStatus.ERROR.value
    
    def test_verification_large_epsilon(self, engine, simple_model, sample_input):
        """Test verification with large epsilon (likely to falsify)"""
        config = VerificationConfig(epsilon=1.0, norm="inf", timeout=30)
        
        result = engine.verify(simple_model, sample_input, config)
        
        # Should complete without error
        assert result.status != VerificationStatus.ERROR.value
        assert result.verification_time > 0
        
        # Large epsilon likely to be falsified
        if result.status == VerificationStatus.FALSIFIED.value:
            assert not result.verified
    
    def test_verification_different_norms(self, engine, simple_model, sample_input):
        """Test verification with different norms"""
        for norm in ["inf", "2"]:
            config = VerificationConfig(epsilon=0.1, norm=norm, timeout=30)
            result = engine.verify(simple_model, sample_input, config)
            
            assert result.status != VerificationStatus.ERROR.value
            assert result.additional_info['perturbation_norm'] == norm
    
    def test_verification_different_bound_methods(self, engine, simple_model, sample_input):
        """Test verification with different bound methods"""
        for method in ["IBP", "CROWN"]:  # Skip alpha-CROWN for speed
            config = VerificationConfig(
                epsilon=0.1, 
                norm="inf", 
                bound_method=method,
                timeout=30
            )
            result = engine.verify(simple_model, sample_input, config)
            
            assert result.status != VerificationStatus.ERROR.value
            assert result.additional_info['bound_method'] == method
    
    def test_verification_timeout(self, engine, simple_model, sample_input):
        """Test verification timeout handling"""
        config = VerificationConfig(epsilon=0.1, norm="inf", timeout=1)  # Very short timeout
        
        start_time = time.time()
        result = engine.verify(simple_model, sample_input, config)
        elapsed = time.time() - start_time
        
        # Should respect timeout (with some tolerance)
        assert elapsed < 5.0  # Should not take much longer than timeout
        assert result.verification_time is not None


class TestVerificationEngineFactory:
    """Test verification engine factory functions"""
    
    def test_create_verification_engine(self):
        """Test engine creation via factory"""
        engine = create_verification_engine(device=DEFAULT_DEVICE)
        
        assert engine is not None
        assert hasattr(engine, 'verify')
        assert hasattr(engine, 'get_capabilities')
    
    def test_engine_device_setting(self):
        """Test device setting in engine creation"""
        engine = create_verification_engine(device=DEFAULT_DEVICE)
        assert engine.device.type == DEFAULT_DEVICE


class TestModelInfo:
    """Test model information extraction"""
    
    def test_model_info_extraction(self):
        """Test extracting information from models"""
        from core.verification.base import ModelInfo
        
        model = create_test_model("tiny")
        input_shape = (10,)
        
        info = ModelInfo.from_model(model, input_shape)
        
        assert info.input_shape == input_shape
        assert info.num_parameters > 0
        assert len(info.layer_types) > 0
        assert 'Linear' in info.layer_types
        assert 'ReLU' in info.activation_functions


# Integration test for verification workflow
def test_complete_verification_workflow():
    """Test complete verification workflow"""
    # Create components
    engine = AlphaBetaCrownEngine(device=DEFAULT_DEVICE)
    model = create_test_model("tiny")
    input_sample = create_sample_input("tiny")
    
    # Test multiple scenarios
    test_configs = [
        VerificationConfig(epsilon=0.01, norm="inf", bound_method="IBP"),
        VerificationConfig(epsilon=0.1, norm="2", bound_method="CROWN"),
        VerificationConfig(epsilon=0.5, norm="inf", bound_method="IBP"),
    ]
    
    results = []
    for config in test_configs:
        result = engine.verify(model, input_sample, config)
        results.append(result)
        
        # Each result should be valid
        assert isinstance(result, VerificationResult)
        assert result.status.value in [s.value for s in VerificationStatus]
        assert result.verification_time > 0
    
    # Should have completed all tests
    assert len(results) == len(test_configs)
    
    # No errors should occur
    error_results = [r for r in results if r.status == VerificationStatus.ERROR.value]
    assert len(error_results) == 0, f"Found {len(error_results)} errors in verification workflow"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])