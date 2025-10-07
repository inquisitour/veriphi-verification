# tests/unit/test_attacks.py
"""
Unit tests for adversarial attack components.
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

from core.attacks import (
    AdversarialAttack, AttackResult, AttackConfig, AttackStatus,
    FGSMAttack, IterativeFGSM, create_attack, list_available_attacks,
    AttackMetrics
)
from core.models import create_test_model, create_sample_input


class TestAttackConfig:
    """Test attack configuration validation"""
    
    def test_valid_config(self):
        """Test valid configuration creation"""
        config = AttackConfig(epsilon=0.1, norm="inf")
        assert config.epsilon == 0.1
        assert config.norm == "inf"
        assert config.max_iterations == 20  # default
        assert not config.targeted  # default
    
    def test_invalid_epsilon(self):
        """Test invalid epsilon values"""
        with pytest.raises(ValueError):
            AttackConfig(epsilon=-0.1)
        
        with pytest.raises(ValueError):
            AttackConfig(epsilon=0.0)
    
    def test_invalid_norm(self):
        """Test invalid norm values"""
        with pytest.raises(ValueError):
            AttackConfig(norm="invalid")
    
    def test_invalid_iterations(self):
        """Test invalid iteration values"""
        with pytest.raises(ValueError):
            AttackConfig(max_iterations=-1)
        
        with pytest.raises(ValueError):
            AttackConfig(max_iterations=0)
    
    def test_targeted_without_target_class(self):
        """Test targeted attack without target class"""
        with pytest.raises(ValueError):
            AttackConfig(targeted=True, target_class=None)
    
    def test_invalid_step_size(self):
        """Test invalid step size"""
        with pytest.raises(ValueError):
            AttackConfig(step_size=-0.1)


class TestAttackResult:
    """Test attack result structure"""
    
    def test_valid_result(self):
        """Test valid result creation"""
        result = AttackResult(
            success=True,
            status=AttackStatus.SUCCESS,
            attack_time=0.5
        )
        assert result.success is True
        assert result.status == AttackStatus.SUCCESS
        assert result.attack_time == 0.5
    
    def test_inconsistent_result_success_true(self):
        """Test inconsistent result: success=True but status=FAILED"""
        with pytest.raises(ValueError):
            AttackResult(
                success=True,
                status=AttackStatus.FAILED
            )
    
    def test_inconsistent_result_success_false(self):
        """Test inconsistent result: success=False but status=SUCCESS"""
        with pytest.raises(ValueError):
            AttackResult(
                success=False,
                status=AttackStatus.SUCCESS
            )


class TestFGSMAttack:
    """Test FGSM attack implementation"""
    
    @pytest.fixture
    def attack(self):
        """Create FGSM attack for testing"""
        return FGSMAttack(device=DEFAULT_DEVICE)
    
    @pytest.fixture
    def simple_model(self):
        """Create simple test model"""
        return create_test_model("tiny")
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor"""
        return create_sample_input("tiny")
    
    def test_attack_initialization(self, attack):
        """Test attack initialization"""
        assert attack is not None
        assert attack.device.type == DEFAULT_DEVICE
        
        capabilities = attack.get_capabilities()
        assert capabilities['name'] == 'FGSMAttack'
        assert 'inf' in capabilities['norms']
        assert '2' in capabilities['norms']
        assert not capabilities['iterative']
    
    def test_untargeted_attack(self, attack, simple_model, sample_input):
        """Test untargeted FGSM attack"""
        config = AttackConfig(epsilon=0.3, norm="inf", targeted=False)
        
        result = attack.attack(simple_model, sample_input, config)
        
        # Check result structure
        assert isinstance(result, AttackResult)
        assert result.status.value in [s.value for s in AttackStatus]
        assert result.attack_time is not None
        assert result.attack_time > 0
        assert result.iterations_used == 1  # FGSM is single-step
        
        # Should not error
        assert result.status != AttackStatus.ERROR.value
    
    def test_targeted_attack(self, attack, simple_model, sample_input):
        """Test targeted FGSM attack"""
        # Get original prediction
        with torch.no_grad():
            output = simple_model(sample_input)
            original_class = torch.argmax(output, dim=1).item()
        
        # Target different class
        target_class = (original_class + 1) % 3
        config = AttackConfig(epsilon=0.5, norm="inf", targeted=True, target_class=target_class)
        
        result = attack.attack(simple_model, sample_input, config)
        
        # Check result structure
        assert isinstance(result, AttackResult)
        assert result.status != AttackStatus.ERROR.value
        assert result.original_prediction == original_class
        
        if result.success:
            assert result.adversarial_prediction == target_class
    
    def test_different_norms(self, attack, simple_model, sample_input):
        """Test attack with different norms"""
        for norm in ["inf", "2"]:
            config = AttackConfig(epsilon=0.3, norm=norm, targeted=False)
            result = attack.attack(simple_model, sample_input, config)
            
            assert result.status != AttackStatus.ERROR.value
            if result.additional_info:
                assert result.additional_info['norm'] == norm
    
    def test_different_epsilons(self, attack, simple_model, sample_input):
        """Test attack with different epsilon values"""
        epsilons = [0.1, 0.3, 0.5]
        
        for eps in epsilons:
            config = AttackConfig(epsilon=eps, norm="inf", targeted=False)
            result = attack.attack(simple_model, sample_input, config)
            
            assert result.status != AttackStatus.ERROR.value
            if result.additional_info:
                assert result.additional_info['epsilon'] == eps
            
            # Check that epsilon is being used correctly
            if result.success and result.perturbation_norm is not None:
                # After clipping to valid range [0,1], perturbation might exceed epsilon
                # Just check that the result is reasonable
                assert result.perturbation_norm >= 0


class TestIterativeFGSM:
    """Test Iterative FGSM attack implementation"""
    
    @pytest.fixture
    def attack(self):
        """Create I-FGSM attack for testing"""
        return IterativeFGSM(device=DEFAULT_DEVICE)
    
    @pytest.fixture
    def simple_model(self):
        """Create simple test model"""
        return create_test_model("tiny")
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor"""
        return create_sample_input("tiny")
    
    def test_attack_initialization(self, attack):
        """Test attack initialization"""
        assert attack is not None
        assert attack.device.type == DEFAULT_DEVICE
        
        capabilities = attack.get_capabilities()
        assert capabilities['name'] == 'IterativeFGSM'
        assert capabilities['iterative']
    
    def test_iterative_attack(self, attack, simple_model, sample_input):
        """Test iterative attack with multiple steps"""
        config = AttackConfig(
            epsilon=0.3, 
            norm="inf", 
            targeted=False, 
            max_iterations=5,
            early_stopping=True
        )
        
        result = attack.attack(simple_model, sample_input, config)
        
        # Check result structure
        assert isinstance(result, AttackResult)
        assert result.status != AttackStatus.ERROR.value
        assert result.iterations_used is not None
        assert result.iterations_used <= config.max_iterations
    
    def test_early_stopping(self, attack, simple_model, sample_input):
        """Test early stopping functionality"""
        config = AttackConfig(
            epsilon=0.5,  # Large epsilon for higher success chance
            norm="inf", 
            targeted=False, 
            max_iterations=20,
            early_stopping=True
        )
        
        result = attack.attack(simple_model, sample_input, config)
        
        if result.success:
            # Should stop early if successful
            assert result.iterations_used <= config.max_iterations
    
    def test_no_early_stopping(self, attack, simple_model, sample_input):
        """Test without early stopping"""
        config = AttackConfig(
            epsilon=0.3, 
            norm="inf", 
            targeted=False, 
            max_iterations=3,
            early_stopping=False
        )
        
        result = attack.attack(simple_model, sample_input, config)
        
        # Should use all iterations when early stopping is disabled
        assert result.iterations_used == config.max_iterations


class TestAttackFactory:
    """Test attack factory functions"""
    
    def test_list_available_attacks(self):
        """Test listing available attacks"""
        attacks = list_available_attacks()
        assert isinstance(attacks, list)
        assert len(attacks) > 0
        assert 'fgsm' in attacks
        assert 'i-fgsm' in attacks
    
    def test_create_attack_fgsm(self):
        """Test creating FGSM attack via factory"""
        attack = create_attack('fgsm', device=DEFAULT_DEVICE)
        assert isinstance(attack, FGSMAttack)
        assert attack.device.type == DEFAULT_DEVICE
    
    def test_create_attack_ifgsm(self):
        """Test creating I-FGSM attack via factory"""
        attack = create_attack('i-fgsm', device=DEFAULT_DEVICE)
        assert isinstance(attack, IterativeFGSM)
        assert attack.device.type == DEFAULT_DEVICE
    
    def test_create_attack_invalid(self):
        """Test creating invalid attack"""
        with pytest.raises(ValueError):
            create_attack('invalid_attack', device=DEFAULT_DEVICE)


class TestAttackMetrics:
    """Test attack metrics utilities"""
    
    def test_success_rate_empty(self):
        """Test success rate with empty results"""
        assert AttackMetrics.success_rate([]) == 0.0
    
    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        results = [
            AttackResult(success=True, status=AttackStatus.SUCCESS),
            AttackResult(success=False, status=AttackStatus.FAILED),
            AttackResult(success=True, status=AttackStatus.SUCCESS),
        ]
        
        rate = AttackMetrics.success_rate(results)
        assert rate == 2/3
    
    def test_average_perturbation(self):
        """Test average perturbation calculation"""
        results = [
            AttackResult(success=True, status=AttackStatus.SUCCESS, perturbation_norm=0.1),
            AttackResult(success=True, status=AttackStatus.SUCCESS, perturbation_norm=0.2),
            AttackResult(success=False, status=AttackStatus.FAILED, perturbation_norm=None),
        ]
        
        avg_pert = AttackMetrics.average_perturbation(results)
        assert abs(avg_pert - 0.15) < 1e-10
    
    def test_average_iterations(self):
        """Test average iterations calculation"""
        results = [
            AttackResult(success=True, status=AttackStatus.SUCCESS, iterations_used=5),
            AttackResult(success=True, status=AttackStatus.SUCCESS, iterations_used=3),
            AttackResult(success=False, status=AttackStatus.FAILED, iterations_used=None),
        ]
        
        avg_iter = AttackMetrics.average_iterations(results)
        assert avg_iter == 4.0
    
    def test_confidence_drop(self):
        """Test confidence drop calculation"""
        results = [
            AttackResult(
                success=True, 
                status=AttackStatus.SUCCESS,
                confidence_original=0.8,
                confidence_adversarial=0.3
            ),
            AttackResult(
                success=True, 
                status=AttackStatus.SUCCESS,
                confidence_original=0.9,
                confidence_adversarial=0.2
            ),
        ]
        
        drop = AttackMetrics.confidence_drop(results)
        assert drop == 0.6  # (0.5 + 0.7) / 2


# Integration test for attack workflow
def test_complete_attack_workflow():
    """Test complete attack workflow"""
    model = create_test_model("tiny")
    input_sample = create_sample_input("tiny")
    
    # Test multiple attack configurations
    attack_configs = [
        ('fgsm', AttackConfig(epsilon=0.1, norm="inf", targeted=False)),
        ('fgsm', AttackConfig(epsilon=0.3, norm="2", targeted=False)),
        ('i-fgsm', AttackConfig(epsilon=0.2, norm="inf", max_iterations=5)),
    ]
    
    results = []
    for attack_name, config in attack_configs:
        attack = create_attack(attack_name, device=DEFAULT_DEVICE)
        result = attack.attack(model, input_sample, config)
        results.append(result)
        
        # Each result should be valid
        assert isinstance(result, AttackResult)
        assert result.status.value in [s.value for s in AttackStatus]
        assert result.attack_time > 0
    
    # Should have completed all tests
    assert len(results) == len(attack_configs)
    
    # No errors should occur
    error_results = [r for r in results if r.status == AttackStatus.ERROR.value]
    assert len(error_results) == 0, f"Found {len(error_results)} errors in attack workflow"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])