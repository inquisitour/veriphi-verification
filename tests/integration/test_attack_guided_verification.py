# tests/integration/test_attack_guided_verification.py
"""
Integration tests for the complete attack-guided verification system.
"""

import pytest
import torch
import sys
import os
import time

DEFAULT_DEVICE = os.environ.get("VERIPHI_DEVICE", "cpu")

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core import (
    VeriphiCore, create_core_system, quick_robustness_check,
    VerificationStatus, AttackStatus
)
from core.verification.attack_guided import AttackGuidedEngine
from core.models import create_test_model, create_sample_input, MODEL_CONFIGS


class TestAttackGuidedEngine:
    """Test attack-guided verification engine integration"""
    
    @pytest.fixture
    def engine(self):
        """Create attack-guided engine for testing"""
        return AttackGuidedEngine(device=DEFAULT_DEVICE, attack_timeout=5.0)
    
    @pytest.fixture
    def simple_model(self):
        """Create simple test model"""
        return create_test_model("tiny")
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor"""
        return create_sample_input("tiny")
    
    def test_engine_initialization(self, engine):
        """Test attack-guided engine initialization"""
        assert engine is not None
        assert str(engine.device) == DEFAULT_DEVICE
        assert engine.attack_timeout == 5.0
        assert len(engine.attacks) > 0
        
        capabilities = engine.get_capabilities()
        assert capabilities['verification_strategy'] == 'attack-guided'
        assert 'attack_methods' in capabilities
        assert capabilities['fast_falsification'] is True
    
    def test_verify_with_attacks_small_epsilon(self, engine, simple_model, sample_input):
        """Test attack-guided verification with small epsilon"""
        from core.verification import VerificationConfig
        
        config = VerificationConfig(epsilon=0.01, norm="inf", timeout=20)
        result = engine.verify_with_attacks(simple_model, sample_input, config)
        
        # Check result structure
        assert result.status.value in [s.value for s in VerificationStatus]
        assert result.verification_time > 0
        assert result.additional_info is not None
        assert result.additional_info['verification_method'] == 'attack-guided'
        
        # Small epsilon should likely be verified or at least not error
        assert result.status != VerificationStatus.ERROR.value
    
    def test_verify_with_attacks_large_epsilon(self, engine, simple_model, sample_input):
        """Test attack-guided verification with large epsilon"""
        from core.verification import VerificationConfig
        
        config = VerificationConfig(epsilon=0.5, norm="inf", timeout=20)
        result = engine.verify_with_attacks(simple_model, sample_input, config)
        
        # Large epsilon should likely be falsified
        assert result.status.value in [
            VerificationStatus.VERIFIED.value,
            VerificationStatus.FALSIFIED.value
        ]
        
        # Check if attack phase was attempted
        assert 'attack_phase_completed' in result.additional_info or \
               'phase_completed' in result.additional_info
    
    def test_attack_phase_timeout(self, simple_model, sample_input):
        """Test attack phase timeout handling"""
        engine = AttackGuidedEngine(device=DEFAULT_DEVICE, attack_timeout=1.0)  # Very short timeout
        from core.verification import VerificationConfig
        
        config = VerificationConfig(epsilon=0.3, norm="inf", timeout=15)
        
        start_time = time.time()
        result = engine.verify_with_attacks(simple_model, sample_input, config)
        attack_phase_time = time.time() - start_time
        
        # Attack phase should respect timeout
        assert attack_phase_time < 5.0  # Should not take much longer than timeout
        assert result.verification_time > 0
    
    def test_batch_verification(self, engine, simple_model):
        """Test batch verification with attack-guided engine"""
        from core.verification import VerificationConfig
        
        # Create batch of inputs
        batch_size = 3
        input_samples = torch.stack([create_sample_input("tiny") for _ in range(batch_size)])
        
        config = VerificationConfig(epsilon=0.2, norm="inf", timeout=10)
        results = engine.verify_batch_with_attacks(simple_model, input_samples, config)
        
        assert len(results) == batch_size
        for result in results:
            assert isinstance(result, type(results[0]))  # Same type
            assert result.verification_time > 0
            assert result.status != VerificationStatus.ERROR.value


class TestVeriphiCore:
    """Test the main VeriphiCore interface"""
    
    @pytest.fixture
    def core_with_attacks(self):
        """Create core system with attacks enabled"""
        return VeriphiCore(use_attacks=True, device=DEFAULT_DEVICE)
    
    @pytest.fixture
    def core_without_attacks(self):
        """Create core system without attacks"""
        return VeriphiCore(use_attacks=False, device=DEFAULT_DEVICE)
    
    @pytest.fixture
    def simple_model(self):
        """Create simple test model"""
        return create_test_model("tiny")
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor"""
        return create_sample_input("tiny")
    
    def test_core_initialization_with_attacks(self, core_with_attacks):
        """Test core system initialization with attacks"""
        assert core_with_attacks.use_attacks is True
        assert hasattr(core_with_attacks.engine, 'verify_with_attacks')
        
        capabilities = core_with_attacks.get_capabilities()
        assert capabilities['attack_support'] is True
        assert len(capabilities['available_attacks']) > 0
    
    def test_core_initialization_without_attacks(self, core_without_attacks):
        """Test core system initialization without attacks"""
        assert core_without_attacks.use_attacks is False
        assert hasattr(core_without_attacks.engine, 'verify')
        
        capabilities = core_without_attacks.get_capabilities()
        assert capabilities['attack_support'] is False
    
    def test_verify_robustness_with_attacks(self, core_with_attacks, simple_model, sample_input):
        """Test robustness verification with attacks enabled"""
        result = core_with_attacks.verify_robustness(
            simple_model, sample_input, 
            epsilon=0.1, norm="inf", timeout=20
        )
        
        assert result.status.value in [s.value for s in VerificationStatus]
        assert result.verification_time > 0
        
        # Should use attack-guided method
        if result.additional_info:
            assert result.additional_info.get('verification_method') == 'attack-guided'
    
    def test_verify_robustness_without_attacks(self, core_without_attacks, simple_model, sample_input):
        """Test robustness verification without attacks"""
        result = core_without_attacks.verify_robustness(
            simple_model, sample_input,
            epsilon=0.1, norm="inf", timeout=20
        )
        
        assert result.status.value in [s.value for s in VerificationStatus]
        assert result.verification_time > 0
        
        # Should not use attack-guided method
        if result.additional_info:
            assert result.additional_info.get('verification_method') != 'attack-guided'
    
    def test_attack_model_directly(self, core_with_attacks, simple_model, sample_input):
        """Test direct attack functionality"""
        result = core_with_attacks.attack_model(
            simple_model, sample_input,
            attack_name='fgsm', epsilon=0.3, norm='inf'
        )
        
        assert result.status.value in [s.value for s in AttackStatus]
        assert result.attack_time > 0
    
    def test_batch_verification(self, core_with_attacks, simple_model):
        """Test batch verification through core interface"""
        # Create batch of inputs
        batch_size = 2
        input_samples = torch.stack([create_sample_input("tiny") for _ in range(batch_size)])
        
        results = core_with_attacks.verify_batch(
            simple_model, input_samples,
            epsilon=0.1, norm="inf", timeout=20
        )
        
        assert len(results) == batch_size
        for result in results:
            assert result.verification_time > 0
            assert result.status != VerificationStatus.ERROR.value
    
    def test_evaluate_robustness(self, core_with_attacks, simple_model):
        """Test comprehensive robustness evaluation"""
        # Create test inputs
        test_inputs = torch.stack([create_sample_input("tiny") for _ in range(2)])
        
        evaluation = core_with_attacks.evaluate_robustness(
            simple_model, test_inputs,
            epsilons=[0.05, 0.1, 0.2],
            norm="inf"
        )
        
        assert isinstance(evaluation, dict)
        assert len(evaluation) == 3  # Three epsilon values
        
        for eps, stats in evaluation.items():
            assert 'verification_rate' in stats
            assert 'falsification_rate' in stats
            assert 'total_samples' in stats
            assert 'average_time' in stats
            assert stats['total_samples'] == 2
            assert 0 <= stats['verification_rate'] <= 1
            assert 0 <= stats['falsification_rate'] <= 1


class TestDifferentModelArchitectures:
    """Test verification with different model architectures"""
    
    @pytest.fixture
    def core(self):
        """Create core system for testing"""
        return create_core_system(use_attacks=True, device=DEFAULT_DEVICE)
    
    @pytest.mark.parametrize("model_type", ["tiny", "linear"])
    def test_different_models(self, core, model_type):
        """Test verification with different model types"""
        model = create_test_model(model_type)
        input_sample = create_sample_input(model_type)
        
        result = core.verify_robustness(
            model, input_sample,
            epsilon=0.1, norm="inf", timeout=15
        )
        
        assert result.status != VerificationStatus.ERROR.value
        assert result.verification_time > 0
    
    def test_model_configs(self, core):
        """Test verification with predefined model configurations"""
        # Test a few model configurations
        configs_to_test = ["toy_problem", "mnist_simple"]
        
        for config_name in configs_to_test:
            if config_name in MODEL_CONFIGS:
                from core.models import create_model_from_config
                model = create_model_from_config(config_name)
                
                # Determine input based on config
                if "toy" in config_name:
                    input_sample = create_sample_input("tiny")
                else:
                    input_sample = create_sample_input("linear")
                
                result = core.verify_robustness(
                    model, input_sample,
                    epsilon=0.05, norm="inf", timeout=15
                )
                
                assert result.status != VerificationStatus.ERROR.value


class TestPerformanceCharacteristics:
    """Test performance characteristics of the verification system"""
    
    @pytest.fixture
    def core(self):
        """Create core system for testing"""
        return create_core_system(use_attacks=True, device=DEFAULT_DEVICE)
    
    def test_verification_time_scaling(self, core):
        """Test that verification time scales reasonably"""
        model = create_test_model("tiny")
        input_sample = create_sample_input("tiny")
        
        # Test different epsilon values
        epsilons = [0.01, 0.1, 0.5]
        times = []
        
        for eps in epsilons:
            start_time = time.time()
            result = core.verify_robustness(
                model, input_sample,
                epsilon=eps, norm="inf", timeout=10
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            # Verification should complete quickly for small models
            assert elapsed < 5.0
            assert result.status != VerificationStatus.ERROR.value
        
        # All times should be reasonable
        assert all(t > 0 for t in times)
    
    def test_attack_vs_formal_verification_timing(self):
        """Test timing comparison between attack-guided and pure formal verification"""
        model = create_test_model("tiny")
        input_sample = create_sample_input("tiny")
        
        # Attack-guided verification
        core_with_attacks = create_core_system(use_attacks=True, device=DEFAULT_DEVICE)
        start_time = time.time()
        result_attacks = core_with_attacks.verify_robustness(
            model, input_sample, epsilon=0.3, timeout=15
        )
        time_with_attacks = time.time() - start_time
        
        # Pure formal verification
        core_without_attacks = create_core_system(use_attacks=False, device=DEFAULT_DEVICE)
        start_time = time.time()
        result_formal = core_without_attacks.verify_robustness(
            model, input_sample, epsilon=0.3, timeout=15
        )
        time_formal = time.time() - start_time
        
        # Both should complete successfully
        assert result_attacks.status != VerificationStatus.ERROR.value
        assert result_formal.status != VerificationStatus.ERROR.value
        
        # Times should be reasonable
        assert time_with_attacks > 0
        assert time_formal > 0


class TestFactoryFunctions:
    """Test factory functions and convenience methods"""
    
    def test_create_core_system(self):
        """Test core system factory function"""
        # With attacks
        core_attacks = create_core_system(use_attacks=True, device=DEFAULT_DEVICE)
        assert core_attacks.use_attacks is True
        
        # Without attacks
        core_no_attacks = create_core_system(use_attacks=False, device=DEFAULT_DEVICE)
        assert core_no_attacks.use_attacks is False
    
    def test_quick_robustness_check(self):
        """Test quick robustness check convenience function"""
        model = create_test_model("tiny")
        input_sample = create_sample_input("tiny")
        
        # Small epsilon should likely be robust
        is_robust = quick_robustness_check(model, input_sample, epsilon=0.01)
        assert isinstance(is_robust, bool)
        
        # Large epsilon should likely not be robust  
        is_robust_large = quick_robustness_check(model, input_sample, epsilon=1.0)
        assert isinstance(is_robust_large, bool)


# End-to-end workflow test
def test_complete_end_to_end_workflow():
    """Test complete end-to-end workflow"""
    print("\n🚀 Running complete end-to-end workflow test...")
    
    # 1. Create system
    core = create_core_system(use_attacks=True, device=DEFAULT_DEVICE)
    
    # 2. Create model and input
    model = create_test_model("tiny")
    input_sample = create_sample_input("tiny")
    
    # 3. Test individual attack
    attack_result = core.attack_model(model, input_sample, attack_name='fgsm', epsilon=0.3)
    assert attack_result.status in [s.value for s in AttackStatus]
    
    # 4. Test verification with different epsilons
    test_epsilons = [0.05, 0.1, 0.3]
    verification_results = []
    
    for eps in test_epsilons:
        result = core.verify_robustness(model, input_sample, epsilon=eps, timeout=15)
        verification_results.append(result)
        assert result.status != VerificationStatus.ERROR.value
    
    # 5. Test batch verification
    batch_inputs = torch.stack([create_sample_input("tiny") for _ in range(2)])
    batch_results = core.verify_batch(model, batch_inputs, epsilon=0.1, timeout=20)
    assert len(batch_results) == 2
    
    # 6. Test robustness evaluation
    evaluation = core.evaluate_robustness(
        model, batch_inputs,
        epsilons=[0.05, 0.2],
        norm="inf"
    )
    assert len(evaluation) == 2
    
    print("✅ Complete end-to-end workflow test passed!")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])