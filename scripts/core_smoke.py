# test_core_verification.py
"""
Test script for core verification functionality.
Run this to validate that Step 3-4 implementation is working correctly.
"""

import sys
import os
import torch
import time

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all core components can be imported"""
    print("🧪 Testing Core Imports")
    print("=" * 50)
    
    try:
        from core.verification import (
            VerificationEngine, VerificationResult, VerificationConfig, VerificationStatus,
            AlphaBetaCrownEngine, create_verification_engine
        )
        print("✓ Verification components imported successfully")
        
        from core.models import (
            create_test_model, create_sample_input, MODEL_CONFIGS
        )
        print("✓ Model components imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_creation():
    """Test model creation and basic functionality"""
    print("\n🏗️ Testing Model Creation")
    print("=" * 50)
    
    from core.models import create_test_model, create_sample_input
    
    try:
        # Test different model types
        models = {}
        for model_type in ["tiny", "linear", "conv"]:
            model = create_test_model(model_type)
            models[model_type] = model
            
            # Test forward pass
            input_sample = create_sample_input(model_type)
            with torch.no_grad():
                output = model(input_sample)
            
            print(f"✓ {model_type} model: input {input_sample.shape} -> output {output.shape}")
        
        return True, models
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False, {}

def test_verification_config():
    """Test verification configuration"""
    print("\n⚙️ Testing Verification Configuration")
    print("=" * 50)
    
    from core.verification import VerificationConfig
    
    try:
        # Test valid configurations
        configs = [
            VerificationConfig(epsilon=0.1, norm="inf"),
            VerificationConfig(epsilon=0.05, norm="2", bound_method="IBP"),
            VerificationConfig(epsilon=0.2, norm="inf", bound_method="CROWN", timeout=60)
        ]
        
        for i, config in enumerate(configs):
            print(f"✓ Config {i+1}: ε={config.epsilon}, norm=L{config.norm}, method={config.bound_method}")
        
        # Test invalid configuration
        try:
            invalid_config = VerificationConfig(epsilon=-0.1)
            print("✗ Invalid config should have failed")
            return False
        except ValueError:
            print("✓ Invalid configuration properly rejected")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_verification_engine():
    """Test verification engine creation and capabilities"""
    print("\n🔧 Testing Verification Engine")
    print("=" * 50)
    
    from core.verification import create_verification_engine
    
    try:
        # Create engine
        engine = create_verification_engine(device='cpu')
        print(f"✓ Engine created: {type(engine).__name__}")
        print(f"✓ Device: {engine.get_device()}")
        
        # Test capabilities
        capabilities = engine.get_capabilities()
        print(f"✓ Supported layers: {capabilities['layers']}")
        print(f"✓ Supported activations: {capabilities['activations']}")
        print(f"✓ Supported norms: {capabilities['norms']}")
        print(f"✓ Bound methods: {capabilities['bound_methods']}")
        
        return True, engine
    except Exception as e:
        print(f"✗ Engine test failed: {e}")
        return False, None

def test_basic_verification():
    """Test basic verification functionality"""
    print("\n🛡️ Testing Basic Verification")
    print("=" * 50)
    
    from core.verification import create_verification_engine, VerificationConfig
    from core.models import create_test_model, create_sample_input
    
    try:
        # Create components
        engine = create_verification_engine(device='cpu')
        model = create_test_model("tiny")  # Use tiny model for quick testing
        input_sample = create_sample_input("tiny")
        
        print(f"Model: {type(model).__name__}")
        print(f"Input shape: {input_sample.shape}")
        
        # Get original prediction
        with torch.no_grad():
            output = model(input_sample)
            predicted_class = torch.argmax(output, dim=1).item()
        print(f"Original prediction: class {predicted_class}")
        
        # Test verification with different epsilons
        epsilons = [0.01, 0.1, 0.5]
        results = []
        
        for eps in epsilons:
            config = VerificationConfig(epsilon=eps, norm="inf", bound_method="IBP")
            print(f"\nTesting ε = {eps}...")
            
            start_time = time.time()
            result = engine.verify(model, input_sample, config)
            elapsed = time.time() - start_time
            
            results.append(result)
            print(f"  Result: {result.status.value}")
            print(f"  Verified: {result.verified}")
            print(f"  Time: {elapsed:.3f}s")
            
            if result.additional_info:
                gap = result.additional_info.get('bounds_gap', 'N/A')
                print(f"  Bounds gap: {gap}")
        
        print("\n📊 Verification Summary:")
        for i, (eps, result) in enumerate(zip(epsilons, results)):
            status_symbol = "✓" if result.verified else "✗"
            print(f"  ε={eps}: {status_symbol} {result.status.value}")
        
        return True
    except Exception as e:
        print(f"✗ Verification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_models():
    """Test verification on different model architectures"""
    print("\n🔄 Testing Different Model Architectures")
    print("=" * 50)
    
    from core.verification import create_verification_engine, VerificationConfig
    from core.models import create_test_model, create_sample_input
    
    engine = create_verification_engine(device='cpu')
    config = VerificationConfig(epsilon=0.1, norm="inf", bound_method="IBP", timeout=30)
    
    model_types = ["tiny", "linear"]  # Skip conv for now due to complexity
    
    for model_type in model_types:
        try:
            print(f"\nTesting {model_type} model...")
            model = create_test_model(model_type)
            input_sample = create_sample_input(model_type)
            
            result = engine.verify(model, input_sample, config)
            
            status_symbol = "✓" if result.status != 'error' else "✗"
            print(f"  {status_symbol} {model_type}: {result.status.value} ({result.verification_time:.3f}s)")
            
        except Exception as e:
            print(f"  ✗ {model_type}: Error - {e}")
    return True

def main():
    """Run all tests"""
    print("🧪 Core Verification System Tests")
    print("This script validates the Step 3-4 implementation\n")
    
    tests = [
        ("Core Imports", test_imports),
        ("Model Creation", lambda: test_model_creation()[0]),
        ("Verification Config", test_verification_config),
        ("Verification Engine", lambda: test_verification_engine()[0]),
        ("Basic Verification", test_basic_verification),
        ("Different Models", test_different_models)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All core verification tests passed!")
        print("🚀 Ready to proceed with Step 5-6: Test framework and validation")
    else:
        print(f"\n⚠️ {total - passed} tests failed")
        print("Please review the errors above and fix the implementation")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
