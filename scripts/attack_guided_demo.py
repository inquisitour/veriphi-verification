# test_attack_guided_verification.py
"""
Comprehensive test script for attack-guided verification functionality.
Tests Adversarial Attacks Integration.
"""

import sys
import os
import torch
import time

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

DEVICE = os.environ.get("VERIPHI_DEVICE", "cpu")

def test_attack_imports():
    """Test attack component imports"""
    print("🗡️ Testing Attack Imports")
    print("=" * 50)
    
    try:
        from core.attacks import (
            AdversarialAttack, AttackResult, AttackConfig, AttackStatus,
            FGSMAttack, IterativeFGSM, create_attack, list_available_attacks
        )
        print("✓ Attack components imported successfully")
        
        available_attacks = list_available_attacks()
        print(f"✓ Available attacks: {available_attacks}")
        
        return True
    except ImportError as e:
        print(f"✗ Attack import failed: {e}")
        return False

def test_attack_creation():
    """Test attack instantiation"""
    print("\n⚔️ Testing Attack Creation")
    print("=" * 50)
    
    from core.attacks import create_attack, FGSMAttack, IterativeFGSM
    
    try:
        # Test factory function
        fgsm = create_attack('fgsm', DEVICE)
        ifgsm = create_attack('i-fgsm', DEVICE)
        
        print(f"✓ FGSM created: {type(fgsm).__name__}")
        print(f"✓ I-FGSM created: {type(ifgsm).__name__}")
        
        # Test capabilities
        fgsm_caps = fgsm.get_capabilities()
        print(f"✓ FGSM capabilities: {fgsm_caps['norms']}, iterative: {fgsm_caps['iterative']}")
        
        ifgsm_caps = ifgsm.get_capabilities()
        print(f"✓ I-FGSM capabilities: {ifgsm_caps['norms']}, iterative: {ifgsm_caps['iterative']}")
        
        return True, [fgsm, ifgsm]
    except Exception as e:
        print(f"✗ Attack creation failed: {e}")
        return False, []

def test_basic_attacks():
    """Test basic attack functionality"""
    print("\n🎯 Testing Basic Attack Functionality")
    print("=" * 50)
    
    from core.attacks import FGSMAttack, AttackConfig
    from core.models import create_test_model, create_sample_input
    
    try:
        # Create components
        attack = FGSMAttack(DEVICE)
        model = create_test_model("tiny")
        input_sample = create_sample_input("tiny")
        
        print(f"Model: {type(model).__name__}")
        print(f"Input shape: {input_sample.shape}")
        
        # Get original prediction
        with torch.no_grad():
            output = model(input_sample)
            original_class = torch.argmax(output, dim=1).item()
        print(f"Original prediction: class {original_class}")
        
        # Test untargeted attack
        config = AttackConfig(epsilon=0.3, norm="inf", targeted=False)
        print(f"\nTesting untargeted FGSM (ε={config.epsilon})...")
        
        result = attack.attack(model, input_sample, config)
        
        print(f"Attack result: {result.status.value}")
        print(f"Success: {result.success}")
        if result.success:
            print(f"  {original_class} -> {result.adversarial_prediction}")
            print(f"  Perturbation norm: {result.perturbation_norm:.6f}")
            print(f"  Confidence drop: {result.additional_info.get('confidence_drop', 0):.3f}")
        
        # Test targeted attack
        target_class = (original_class + 1) % 3  # Different class
        config_targeted = AttackConfig(epsilon=0.5, norm="inf", targeted=True, target_class=target_class)
        print(f"\nTesting targeted FGSM (target: {target_class})...")
        
        result_targeted = attack.attack(model, input_sample, config_targeted)
        print(f"Targeted attack result: {result_targeted.status.value}")
        if result_targeted.success:
            print(f"  Successfully targeted: {original_class} -> {result_targeted.adversarial_prediction}")
        
        return True
    except Exception as e:
        print(f"✗ Basic attack test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_attack_guided_verification():
    """Test attack-guided verification engine"""
    print("\n🛡️ Testing Attack-Guided Verification")
    print("=" * 50)
    
    from core.verification.attack_guided import AttackGuidedEngine
    from core.verification import VerificationConfig
    from core.models import create_test_model, create_sample_input
    
    try:
        # Create components
        engine = AttackGuidedEngine(DEVICE, attack_timeout=5.0)
        model = create_test_model("tiny")
        input_sample = create_sample_input("tiny")
        
        print(f"Engine: {type(engine).__name__}")
        print(f"Attack timeout: {engine.attack_timeout}s")
        
        # Get original prediction
        with torch.no_grad():
            output = model(input_sample)
            original_class = torch.argmax(output, dim=1).item()
        print(f"Original prediction: class {original_class}")
        
        # Test different epsilon values
        epsilons = [0.01, 0.1, 0.5]  # Small, medium, large
        
        for eps in epsilons:
            config = VerificationConfig(
                epsilon=eps, 
                norm="inf", 
                bound_method="IBP",
                timeout=30
            )
            
            print(f"\n🔍 Testing ε = {eps}...")
            start_time = time.time()
            
            result = engine.verify_with_attacks(model, input_sample, config)
            elapsed = time.time() - start_time
            
            print(f"  Result: {result.status.value}")
            print(f"  Verified: {result.verified}")
            print(f"  Time: {elapsed:.3f}s")
            
            if result.additional_info:
                method = result.additional_info.get('verification_method', 'unknown')
                phase = result.additional_info.get('phase_completed', 'unknown')
                falsification_method = result.additional_info.get('falsification_method')
                
                print(f"  Method: {method}")
                print(f"  Phase completed: {phase}")
                
                if falsification_method:
                    print(f"  Falsified by: {falsification_method}")
                    print(f"  Perturbation norm: {result.additional_info.get('perturbation_norm', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"✗ Attack-guided verification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_core_interface():
    """Test enhanced core interface with attacks"""
    print("\n🚀 Testing Enhanced Core Interface")
    print("=" * 50)
    
    from core import VeriphiCore, create_core_system
    from core.models import create_test_model, create_sample_input
    
    try:
        # Create enhanced core system
        core = create_core_system(use_attacks=True, device=DEVICE)
        model = create_test_model("tiny")
        input_sample = create_sample_input("tiny")
        
        print(f"Core system: {type(core).__name__}")
        print(f"Uses attacks: {core.use_attacks}")
        
        # Test verification
        print("\nTesting robustness verification...")
        result = core.verify_robustness(
            model, input_sample, 
            epsilon=0.1, norm="inf", timeout=20
        )
        
        print(f"Verification result: {result.status.value}")
        print(f"Time: {result.verification_time:.3f}s")
        
        # Test direct attack
        print("\nTesting direct attack...")
        attack_result = core.attack_model(
            model, input_sample,
            attack_name="fgsm", epsilon=0.3, norm="inf"
        )
        
        print(f"Attack result: {attack_result.status.value}")
        print(f"Success: {attack_result.success}")
        
        # Test capabilities
        capabilities = core.get_capabilities()
        print(f"\nCapabilities:")
        print(f"  Attack support: {capabilities.get('attack_support', False)}")
        print(f"  Available attacks: {capabilities.get('available_attacks', [])}")
        print(f"  Verification strategy: {capabilities.get('verification_strategy', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"✗ Enhanced core interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_robustness_evaluation():
    """Test comprehensive robustness evaluation"""
    print("\n📊 Testing Robustness Evaluation")
    print("=" * 50)
    
    from core import create_core_system
    from core.models import create_test_model, create_sample_input
    
    try:
        # Create system and test data
        core = create_core_system(use_attacks=True, device=DEVICE)
        model = create_test_model("tiny")
        
        # Create multiple test samples
        test_inputs = torch.stack([create_sample_input("tiny") for _ in range(3)])
        print(f"Test inputs shape: {test_inputs.shape}")
        
        # Run robustness evaluation
        epsilons = [0.05, 0.1, 0.2]
        print(f"Testing epsilons: {epsilons}")
        
        results = core.evaluate_robustness(
            model, test_inputs, 
            epsilons=epsilons, norm="inf"
        )
        
        # Display results
        print("\nRobustness Evaluation Results:")
        for eps, stats in results.items():
            print(f"  ε = {eps}:")
            print(f"    Verification rate: {stats['verification_rate']:.1%}")
            print(f"    Falsification rate: {stats['falsification_rate']:.1%}")
            print(f"    Average time: {stats['average_time']:.3f}s")
        
        return True
    except Exception as e:
        print(f"✗ Robustness evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_attack_methods():
    """Test different attack methods"""
    print("\n⚔️ Testing Different Attack Methods")
    print("=" * 50)
    
    from core.attacks import create_attack, AttackConfig
    from core.models import create_test_model, create_sample_input
    
    try:
        model = create_test_model("tiny")
        input_sample = create_sample_input("tiny")
        
        attack_names = ["fgsm", "i-fgsm"]
        config = AttackConfig(epsilon=0.3, norm="inf", targeted=False, max_iterations=10)
        
        for attack_name in attack_names:
            print(f"\nTesting {attack_name.upper()}...")
            
            attack = create_attack(attack_name, DEVICE)
            result = attack.attack(model, input_sample, config)
            
            print(f"  Status: {result.status.value}")
            print(f"  Success: {result.success}")
            print(f"  Time: {result.attack_time:.3f}s")
            
            if result.success:
                print(f"  Prediction change: {result.original_prediction} -> {result.adversarial_prediction}")
                print(f"  Perturbation norm: {result.perturbation_norm:.6f}")
            
            if hasattr(result, 'iterations_used') and result.iterations_used:
                print(f"  Iterations used: {result.iterations_used}")
        
        return True
    except Exception as e:
        print(f"✗ Different attack methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_attack_vs_verification():
    """Benchmark attack-guided vs pure verification"""
    print("\n⚡ Benchmarking Attack-Guided vs Pure Verification")
    print("=" * 50)
    
    from core.verification import AlphaBetaCrownEngine, VerificationConfig
    from core.verification.attack_guided import AttackGuidedEngine
    from core.models import create_test_model, create_sample_input
    
    try:
        model = create_test_model("tiny")
        input_sample = create_sample_input("tiny")
        
        # Test with large epsilon (likely to be falsified)
        config = VerificationConfig(epsilon=0.5, norm="inf", bound_method="IBP", timeout=30)
        
        # Pure verification
        print("Pure verification:")
        pure_engine = AlphaBetaCrownEngine(DEVICE)
        start_time = time.time()
        pure_result = pure_engine.verify(model, input_sample, config)
        pure_time = time.time() - start_time
        
        print(f"  Result: {pure_result.status.value}")
        print(f"  Time: {pure_time:.3f}s")
        
        # Attack-guided verification
        print("\nAttack-guided verification:")
        guided_engine = AttackGuidedEngine(DEVICE, attack_timeout=2.0)
        start_time = time.time()
        guided_result = guided_engine.verify_with_attacks(model, input_sample, config)
        guided_time = time.time() - start_time
        
        print(f"  Result: {guided_result.status.value}")
        print(f"  Time: {guided_time:.3f}s")
        
        if guided_result.additional_info:
            method = guided_result.additional_info.get('verification_method', 'unknown')
            phase = guided_result.additional_info.get('phase_completed', 'unknown')
            print(f"  Method: {method}")
            print(f"  Phase: {phase}")
        
        # Compare
        if pure_time > 0 and guided_time > 0:
            speedup = pure_time / guided_time
            print(f"\nSpeedup: {speedup:.1f}x faster with attack-guided approach")
        
        return True
    except Exception as e:
        print(f"✗ Benchmark test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all attack-guided verification tests"""
    print("🗡️🛡️ Attack-Guided Verification System Tests")
    
    tests = [
        ("Attack Imports", test_attack_imports),
        ("Attack Creation", lambda: test_attack_creation()[0]),
        ("Basic Attacks", test_basic_attacks),
        ("Attack-Guided Verification", test_attack_guided_verification),
        ("Enhanced Core Interface", test_enhanced_core_interface),
        ("Robustness Evaluation", test_robustness_evaluation),
        ("Different Attack Methods", test_different_attack_methods),
        ("Attack vs Verification Benchmark", benchmark_attack_vs_verification)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            print("\n" + "=" * 70)
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:30}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All attack-guided verification tests passed!")
        print("\n🚀 Key Features Implemented:")
        print("   • Fast adversarial attacks (FGSM, I-FGSM)")
        print("   • Attack-guided verification strategy")
        print("   • Enhanced core interface with attacks")
        print("   • Comprehensive robustness evaluation")
        print("   • Performance optimizations")
    else:
        print(f"\n⚠️ {total - passed} tests failed")
        print("Please review the errors above and fix the implementation")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)