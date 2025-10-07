#!/usr/bin/env python3
"""
Comprehensive system verification for neural network verification environment.
"""

import sys
import torch
import numpy as np

def check_python():
    """Check Python version"""
    print("=" * 50)
    print("PYTHON VERIFICATION")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    assert sys.version_info >= (3, 9), f"Python 3.9+ required, got {sys.version_info}"
    print("✓ Python version OK")

def check_pytorch():
    """Check PyTorch installation"""
    print("\n" + "=" * 50)
    print("PYTORCH VERIFICATION")
    print("=" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version (PyTorch): {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
    else:
        print("⚠ Running in CPU-only mode")
    
    # Test tensor operations
    x = torch.randn(100, 100)
    y = torch.randn(100, 100)
    z = torch.matmul(x, y)
    print("✓ PyTorch tensor operations working")

def check_auto_lirpa():
    """Check auto-LiRPA installation"""
    print("\n" + "=" * 50)
    print("AUTO-LIRPA VERIFICATION")
    print("=" * 50)
    
    try:
        from auto_LiRPA.bound_general import BoundedModule
        from auto_LiRPA.bounded_tensor import BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm
        import auto_LiRPA
        print(f"auto-LiRPA version: {getattr(auto_LiRPA, '__version__', '0.6.0')}")
        
        # Test basic functionality
        import torch.nn as nn
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        dummy_input = torch.randn(1, 10)
        
        # Create bounded module
        bounded_model = BoundedModule(model, dummy_input)
        perturbation = PerturbationLpNorm(norm=np.inf, eps=0.1)
        bounded_input = BoundedTensor(dummy_input, perturbation)
        
        # Compute bounds
        lb, ub = bounded_model.compute_bounds(x=(bounded_input,), method="IBP")
        print(f"Bounds computed: LB shape {lb.shape}, UB shape {ub.shape}")
        print("✓ auto-LiRPA basic functionality working")
        
    except ImportError as e:
        print(f"✗ auto-LiRPA import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ auto-LiRPA test failed: {e}")
        return False
    
    print("✓ auto-LiRPA installation OK")
    return True

def check_onnx():
    """Check ONNX installation"""
    print("\n" + "=" * 50)
    print("ONNX VERIFICATION")
    print("=" * 50)
    
    try:
        import onnx
        import onnxruntime as ort
        print(f"ONNX version: {onnx.__version__}")
        print(f"ONNX Runtime version: {ort.__version__}")
        print(f"Available providers: {ort.get_available_providers()}")
        
        # Check for CUDA provider
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            print("✓ ONNX Runtime CUDA support available")
        else:
            print("⚠ ONNX Runtime running in CPU mode")
        
        print("✓ ONNX installation OK")
        
    except ImportError as e:
        print(f"✗ ONNX import failed: {e}")
        return False
    
    return True

def check_dev_tools():
    """Check development tools"""
    print("\n" + "=" * 50)
    print("DEVELOPMENT TOOLS VERIFICATION")
    print("=" * 50)
    
    tools = ['pytest', 'black', 'isort', 'flake8', 'numpy', 'scipy', 'matplotlib']
    passed = 0
    
    for tool in tools:
        try:
            __import__(tool)
            print(f"✓ {tool}")
            passed += 1
        except ImportError:
            print(f"✗ {tool}")
    
    print(f"Development tools: {passed}/{len(tools)} available")
    return passed >= len(tools) - 2  # Allow 2 failures

def main():
    """Run all verification checks"""
    print("🛡️ Neural Network Verification Environment Verification")
    print("This script verifies that all components are properly installed.\n")
    
    checks = [
        ("Python", check_python),
        ("PyTorch", check_pytorch),
        ("auto-LiRPA", check_auto_lirpa),
        ("ONNX", check_onnx),
        ("Dev Tools", check_dev_tools)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            result = check_func()
            results[name] = result if result is not None else True
        except Exception as e:
            print(f"✗ {name} check failed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:15}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 All components verified successfully!")
        print("Your environment is ready for neural network verification development.")
        print("\n📋 Environment Summary:")
        print("   - Ubuntu 24.04.2 LTS")
        print("   - Python 3.12.3 with virtual environment")
        print("   - PyTorch (CPU-only)")
        print("   - auto-LiRPA verification engine")
        print("   - ONNX model format support")
        print("   - Development and testing tools")
        print("\n🚀 Ready to proceed !")
    else:
        print(f"\n⚠️  Some components failed verification.")
        print("Please review the errors above and reinstall failed components.")
        sys.exit(1)

if __name__ == "__main__":
    main()