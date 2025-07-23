# tests/benchmarks/test_performance.py
"""
Benchmark tests for performance analysis of the verification system.
"""

import pytest
import torch
import time
import sys
import os
from typing import List, Dict, Any
import statistics

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core import create_core_system, VerificationStatus
from core.models import create_test_model, create_sample_input
from core.attacks import create_attack, AttackConfig


class PerformanceBenchmark:
    """Performance benchmark utilities"""
    
    @staticmethod
    def time_function(func, *args, **kwargs):
        """Time a function execution"""
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time
    
    @staticmethod
    def run_multiple_trials(func, num_trials: int, *args, **kwargs):
        """Run function multiple times and collect statistics"""
        times = []
        results = []
        
        for _ in range(num_trials):
            result, elapsed_time = PerformanceBenchmark.time_function(func, *args, **kwargs)
            times.append(elapsed_time)
            results.append(result)
        
        return {
            'times': times,
            'results': results,
            'mean_time': statistics.mean(times),
            'median_time': statistics.median(times),
            'std_time': statistics.stdev(times) if len(times) > 1 else 0.0,
            'min_time': min(times),
            'max_time': max(times)
        }


class TestVerificationPerformance:
    """Benchmark verification performance"""
    
    def test_single_verification_timing(self):
        """Benchmark single verification timing"""
        core = create_core_system(use_attacks=True, device='cpu')
        model = create_test_model("tiny")
        input_sample = create_sample_input("tiny")
        
        def verify_once():
            return core.verify_robustness(model, input_sample, epsilon=0.1, timeout=30)
        
        # Run multiple trials
        benchmark = PerformanceBenchmark.run_multiple_trials(verify_once, num_trials=5)
        
        print(f"\n📊 Single Verification Performance:")
        print(f"   Mean time: {benchmark['mean_time']:.3f}s ± {benchmark['std_time']:.3f}s")
        print(f"   Range: {benchmark['min_time']:.3f}s - {benchmark['max_time']:.3f}s")
        
        # Performance assertions
        assert benchmark['mean_time'] < 2.0, "Verification should complete in under 2 seconds"
        assert all(r.status != VerificationStatus.ERROR.value for r in benchmark['results'])
    
    def test_batch_verification_scaling(self):
        """Benchmark batch verification scaling"""
        core = create_core_system(use_attacks=True, device='cpu')
        model = create_test_model("tiny")
        
        batch_sizes = [1, 2, 4]
        results = {}
        
        for batch_size in batch_sizes:
            input_samples = torch.stack([create_sample_input("tiny") for _ in range(batch_size)])
            
            def verify_batch():
                return core.verify_batch(model, input_samples, epsilon=0.1, timeout=60)
            
            benchmark = PerformanceBenchmark.run_multiple_trials(verify_batch, num_trials=3)
            results[batch_size] = benchmark
            
            print(f"\n📊 Batch Size {batch_size} Performance:")
            print(f"   Mean time: {benchmark['mean_time']:.3f}s")
            print(f"   Time per sample: {benchmark['mean_time']/batch_size:.3f}s")
        
        # Check scaling characteristics
        for batch_size in batch_sizes:
            time_per_sample = results[batch_size]['mean_time'] / batch_size
            assert time_per_sample < 2.0, f"Time per sample should be under 2s for batch size {batch_size}"
    
    def test_epsilon_scaling_performance(self):
        """Benchmark performance across different epsilon values"""
        core = create_core_system(use_attacks=True, device='cpu')
        model = create_test_model("tiny")
        input_sample = create_sample_input("tiny")
        
        epsilons = [0.01, 0.05, 0.1, 0.2, 0.5]
        results = {}
        
        for eps in epsilons:
            def verify_epsilon():
                return core.verify_robustness(model, input_sample, epsilon=eps, timeout=30)
            
            benchmark = PerformanceBenchmark.run_multiple_trials(verify_epsilon, num_trials=3)
            results[eps] = benchmark
            
            print(f"\n📊 Epsilon {eps} Performance:")
            print(f"   Mean time: {benchmark['mean_time']:.3f}s")
            print(f"   Status distribution: {[r.status for r in benchmark['results']]}")
        
        # All should complete quickly
        for eps in epsilons:
            assert results[eps]['mean_time'] < 3.0, f"Verification at ε={eps} should complete in under 3s"


class TestAttackPerformance:
    """Benchmark attack performance"""
    
    def test_attack_timing_comparison(self):
        """Compare timing of different attack methods"""
        model = create_test_model("tiny")
        input_sample = create_sample_input("tiny")
        
        attack_configs = {
            'fgsm': AttackConfig(epsilon=0.3, norm="inf", targeted=False),
            'i-fgsm': AttackConfig(epsilon=0.3, norm="inf", targeted=False, max_iterations=5)
        }
        
        results = {}
        
        for attack_name, config in attack_configs.items():
            attack = create_attack(attack_name, device='cpu')
            
            def run_attack():
                return attack.attack(model, input_sample, config)
            
            benchmark = PerformanceBenchmark.run_multiple_trials(run_attack, num_trials=5)
            results[attack_name] = benchmark
            
            print(f"\n📊 {attack_name.upper()} Attack Performance:")
            print(f"   Mean time: {benchmark['mean_time']:.4f}s")
            print(f"   Success rate: {sum(r.success for r in benchmark['results'])/len(benchmark['results']):.2f}")
        
        # FGSM should be faster than I-FGSM
        assert results['fgsm']['mean_time'] <= results['i-fgsm']['mean_time'], \
            "FGSM should be faster than I-FGSM"
        
        # Both should be very fast
        assert results['fgsm']['mean_time'] < 0.1, "FGSM should complete in under 0.1s"
        assert results['i-fgsm']['mean_time'] < 0.5, "I-FGSM should complete in under 0.5s"
    
    def test_attack_scaling_with_iterations(self):
        """Test how I-FGSM performance scales with iterations"""
        model = create_test_model("tiny")
        input_sample = create_sample_input("tiny")
        
        iteration_counts = [1, 5, 10, 20]
        results = {}
        
        for num_iter in iteration_counts:
            attack = create_attack('i-fgsm', device='cpu')
            config = AttackConfig(epsilon=0.3, norm="inf", max_iterations=num_iter, early_stopping=False)
            
            def run_attack():
                return attack.attack(model, input_sample, config)
            
            benchmark = PerformanceBenchmark.run_multiple_trials(run_attack, num_trials=3)
            results[num_iter] = benchmark
            
            print(f"\n📊 I-FGSM with {num_iter} iterations:")
            print(f"   Mean time: {benchmark['mean_time']:.4f}s")
            print(f"   Time per iteration: {benchmark['mean_time']/num_iter:.4f}s")
        
        # Time should scale roughly linearly with iterations
        time_per_iter = [results[num_iter]['mean_time']/num_iter for num_iter in iteration_counts]
        
        # All should have similar time per iteration
        assert max(time_per_iter) < 0.05, "Time per iteration should be under 0.05s"


class TestAttackGuidedPerformance:
    """Benchmark attack-guided verification performance"""
    
    def test_attack_vs_formal_verification_speedup(self):
        """Measure speedup of attack-guided vs pure formal verification"""
        model = create_test_model("tiny")
        input_sample = create_sample_input("tiny")
        
        # Test with large epsilon (likely to be falsified quickly by attacks)
        epsilon = 0.5
        
        # Attack-guided verification
        core_with_attacks = create_core_system(use_attacks=True, device='cpu')
        def verify_with_attacks():
            return core_with_attacks.verify_robustness(model, input_sample, epsilon=epsilon, timeout=30)
        
        attack_guided_benchmark = PerformanceBenchmark.run_multiple_trials(verify_with_attacks, num_trials=5)
        
        # Pure formal verification
        core_without_attacks = create_core_system(use_attacks=False, device='cpu')
        def verify_without_attacks():
            return core_without_attacks.verify_robustness(model, input_sample, epsilon=epsilon, timeout=30)
        
        formal_benchmark = PerformanceBenchmark.run_multiple_trials(verify_without_attacks, num_trials=5)
        
        print(f"\n📊 Attack-Guided vs Formal Verification Performance:")
        print(f"   Attack-guided mean time: {attack_guided_benchmark['mean_time']:.3f}s")
        print(f"   Formal verification mean time: {formal_benchmark['mean_time']:.3f}s")
        
        if formal_benchmark['mean_time'] > 0:
            speedup = formal_benchmark['mean_time'] / attack_guided_benchmark['mean_time']
            print(f"   Speedup: {speedup:.2f}x")
        
        # Both should complete successfully
        assert all(r.status != VerificationStatus.ERROR.value for r in attack_guided_benchmark['results'])
        assert all(r.status != VerificationStatus.ERROR.value for r in formal_benchmark['results'])
    
    def test_attack_timeout_effectiveness(self):
        """Test effectiveness of attack timeout settings"""
        model = create_test_model("tiny")
        input_sample = create_sample_input("tiny")
        
        timeout_values = [1.0, 2.0, 5.0]
        results = {}
        
        for timeout in timeout_values:
            from core.verification.attack_guided import AttackGuidedEngine
            engine = AttackGuidedEngine(device='cpu', attack_timeout=timeout)
            
            from core.verification import VerificationConfig
            config = VerificationConfig(epsilon=0.3, norm="inf", timeout=30)
            
            def verify_with_timeout():
                return engine.verify_with_attacks(model, input_sample, config)
            
            benchmark = PerformanceBenchmark.run_multiple_trials(verify_with_timeout, num_trials=3)
            results[timeout] = benchmark
            
            print(f"\n📊 Attack timeout {timeout}s Performance:")
            print(f"   Mean total time: {benchmark['mean_time']:.3f}s")
        
        # Longer timeouts should not significantly increase total time for easy cases
        for timeout in timeout_values:
            assert results[timeout]['mean_time'] < 10.0, f"Total time should be reasonable for timeout {timeout}s"


class TestMemoryUsage:
    """Benchmark memory usage characteristics"""
    
    def test_memory_usage_verification(self):
        """Test memory usage during verification"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        core = create_core_system(use_attacks=True, device='cpu')
        model = create_test_model("tiny")
        input_sample = create_sample_input("tiny")
        
        # Run verification
        result = core.verify_robustness(model, input_sample, epsilon=0.1, timeout=30)
        
        # Peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory
        
        print(f"\n💾 Memory Usage:")
        print(f"   Baseline: {baseline_memory:.1f} MB")
        print(f"   Peak: {peak_memory:.1f} MB")
        print(f"   Increase: {memory_increase:.1f} MB")
        print(f"   Reported by verifier: {result.memory_usage:.1f} MB")
        
        # Memory usage should be reasonable
        assert memory_increase < 500, "Memory increase should be under 500 MB for small models"
    
    def test_memory_scaling_with_model_size(self):
        """Test memory scaling with different model sizes"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        core = create_core_system(use_attacks=True, device='cpu')
        model_types = ["tiny", "linear"]
        
        for model_type in model_types:
            model = create_test_model(model_type)
            input_sample = create_sample_input(model_type)
            
            # Run verification
            result = core.verify_robustness(model, input_sample, epsilon=0.1, timeout=30)
            
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_for_model = current_memory - baseline_memory
            
            print(f"\n💾 {model_type} model memory usage: {memory_for_model:.1f} MB")
            
            # Memory should be reasonable for all model types
            assert memory_for_model < 1000, f"Memory usage for {model_type} model should be under 1GB"


def run_comprehensive_benchmark():
    """Run comprehensive benchmark suite"""
    print("🏃‍♂️ Running Comprehensive Performance Benchmark")
    print("=" * 60)
    
    # Model and setup
    core = create_core_system(use_attacks=True, device='cpu')
    model = create_test_model("tiny")
    input_sample = create_sample_input("tiny")
    
    # Benchmark different scenarios
    scenarios = [
        ("Small ε (0.01)", 0.01),
        ("Medium ε (0.1)", 0.1),
        ("Large ε (0.5)", 0.5),
    ]
    
    total_start_time = time.time()
    
    for scenario_name, epsilon in scenarios:
        print(f"\n🎯 {scenario_name}")
        print("-" * 30)
        
        start_time = time.time()
        result = core.verify_robustness(model, input_sample, epsilon=epsilon, timeout=30)
        elapsed_time = time.time() - start_time
        
        print(f"   Result: {result.status}")
        print(f"   Time: {elapsed_time:.3f}s")
        print(f"   Memory: {result.memory_usage:.1f}MB")
        
        if result.additional_info:
            method = result.additional_info.get('verification_method', 'unknown')
            print(f"   Method: {method}")
    
    total_time = time.time() - total_start_time
    print(f"\n⏱️ Total benchmark time: {total_time:.3f}s")
    print("✅ Comprehensive benchmark completed!")


if __name__ == "__main__":
    # Run comprehensive benchmark
    run_comprehensive_benchmark()
    
    # Run pytest benchmarks
    pytest.main([__file__, "-v", "-s"])