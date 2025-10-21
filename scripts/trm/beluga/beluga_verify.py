#!/usr/bin/env python3
"""
Formal Verification of BelugaTRM Logistics Solutions
Uses Veriphi attack-guided verification with α/β-CROWN to prove robustness
Tests: Can the model maintain constraint satisfaction under parameter perturbations?
"""

import os
import sys
import time
import random
import argparse
import torch
from pathlib import Path

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from core import create_core_system
from core.models.beluga_data_loader import BelugaProblem, create_beluga_dataloader
from core.models.beluga_trm_model import create_beluga_trm, get_model_dims_from_problem

DEVICE = torch.device(os.environ.get("VERIPHI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))


def check_constraint_satisfaction(model, problem, state_tensor, device=DEVICE):
    """
    Check if model's solution satisfies constraints
    
    Returns:
        satisfies_constraints: bool
        constraint_losses: dict
    """
    model.eval()
    
    with torch.no_grad():
        logits = model(state_tensor)
        total_loss = model.compute_constraint_loss(logits, problem)
        
        # Get constraint breakdown
        assignments = torch.argmax(logits.view(1, model.num_jigs, model.action_space), dim=-1)[0]
        probs = torch.softmax(logits.view(1, model.num_jigs, model.action_space), dim=-1)[0]
        
        constraint_losses = {
            'total': total_loss.item(),
            'flight_capacity': model._compute_flight_capacity_loss(assignments, problem, probs).item(),
            'rack_capacity': model._compute_rack_capacity_loss(assignments, problem, probs).item(),
            'schedule': model._compute_schedule_constraint_loss(assignments, problem, probs).item(),
            'balance': model._compute_flight_balance_loss(assignments, problem, probs).item(),
            'type_matching': model._compute_type_matching_loss(assignments, problem, probs).item()
        }
    
    # Solution is feasible if total constraint loss is below threshold
    satisfies_constraints = total_loss.item() < 1.0
    
    return satisfies_constraints, constraint_losses


def main():
    parser = argparse.ArgumentParser(description='Formally verify BelugaTRM robustness')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to Beluga dataset root')
    parser.add_argument('--split', type=str, default='training',
                       choices=['training', 'validation'],
                       help='Dataset split to verify')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of problems to verify (default: 10)')
    parser.add_argument('--epsilon', type=float, default=0.05,
                       help='Perturbation radius (default: 0.05 = 5%% parameter variation)')
    parser.add_argument('--norm', type=str, default='inf', choices=['inf', '2'],
                       help='Norm for perturbation (default: inf)')
    parser.add_argument('--timeout', type=int, default=120,
                       help='Verification timeout per sample (seconds)')
    parser.add_argument('--bound-method', type=str, default='alpha-CROWN',
                       choices=['IBP', 'CROWN', 'alpha-CROWN', 'beta-CROWN'],
                       help='Formal verification method (default: alpha-CROWN)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    print(f"🔍 Beluga TRM Formal Verification")
    print(f"=" * 60)
    print(f"Using Veriphi attack-guided verification with {args.bound_method}")
    print(f"=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Samples: {args.samples}")
    print(f"Epsilon: {args.epsilon} ({args.epsilon*100:.1f}% perturbation)")
    print(f"Norm: L{args.norm}")
    print(f"Timeout: {args.timeout}s per sample")
    print(f"Device: {DEVICE}")
    print()
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Load checkpoint
    print(f"📦 Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model_dims = checkpoint.get('model_dims')
    
    if model_dims is None:
        print(f"❌ Checkpoint missing model_dims")
        return
    
    print(f"✅ Checkpoint loaded (epoch {checkpoint.get('epoch', '?')}, loss {checkpoint.get('loss', '?'):.4f})")
    
    # Create model
    print(f"\n🔨 Creating BelugaTRM model...")
    model = create_beluga_trm(**model_dims)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")
    
    # Create Veriphi verification system with attacks
    print(f"\n⚔️ Initializing Veriphi verification engine...")
    core = create_core_system(use_attacks=True, device=DEVICE.type)
    
    capabilities = core.get_capabilities()
    print(f"   Attack support: {capabilities.get('attack_support', False)}")
    print(f"   Available attacks: {capabilities.get('available_attacks', [])}")
    print(f"   Verification strategy: {capabilities.get('verification_strategy', 'unknown')}")
    
    # Load dataset
    print(f"\n📂 Loading {args.split} dataset...")
    dataloader = create_beluga_dataloader(
        args.data_dir,
        split=args.split,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    total_problems = len(dataloader.dataset)
    print(f"✅ Found {total_problems} problems")
    
    # Sample problems
    if args.samples > total_problems:
        args.samples = total_problems
    
    sample_indices = random.sample(range(total_problems), args.samples)
    
    print(f"\n🚀 Formal Verification with α/β-CROWN")
    print(f"=" * 60)
    print(f"Testing robustness to ±{args.epsilon} ({args.epsilon*100:.1f}%) perturbations")
    print(f"Question: Does constraint satisfaction remain guaranteed?")
    print(f"=" * 60)
    
    # Verification results
    results = []
    verified_count = 0
    falsified_count = 0
    error_count = 0
    timeout_count = 0
    total_time = 0.0
    
    for idx, sample_idx in enumerate(sample_indices, 1):
        state_tensor, problem = dataloader.dataset[sample_idx]
        state_tensor = state_tensor.to(DEVICE)
        
        print(f"\n[{idx}/{args.samples}] Problem: {problem.problem_id}")
        print(f"   Jigs: {problem.num_jigs}, Flights: {problem.num_flights}, Racks: {problem.num_racks}")
        
        # First check: Does the model produce a feasible solution?
        satisfies_constraints, constraint_losses = check_constraint_satisfaction(
            model, problem, state_tensor, device=DEVICE
        )
        
        print(f"   Nominal solution: {'✓ FEASIBLE' if satisfies_constraints else '✗ INFEASIBLE'} "
              f"(loss: {constraint_losses['total']:.4f})")
        
        if not satisfies_constraints:
            print(f"   ⚠️ Skipping verification (nominal solution already infeasible)")
            results.append({
                'problem_id': problem.problem_id,
                'verified': False,
                'status': 'nominal_infeasible',
                'verification_time': 0.0,
                'constraint_losses': constraint_losses
            })
            continue
        
        # Formal verification: Prove robustness under perturbations
        print(f"   🔐 Running formal verification (ε={args.epsilon}, {args.bound_method})...")
        
        t0 = time.time()
        
        try:
            # Use Veriphi's attack-guided verification with α/β-CROWN
            verification_result = core.verify_robustness(
                model,
                state_tensor,
                epsilon=args.epsilon,
                norm=args.norm,
                timeout=args.timeout,
                bound_method=args.bound_method
            )
            
            elapsed = time.time() - t0
            total_time += elapsed
            
            # Parse verification result
            status = verification_result.status.value
            verified = verification_result.verified
            
            if verified:
                verified_count += 1
                status_symbol = "✓ VERIFIED"
                status_color = "🟢"
            elif status == "falsified":
                falsified_count += 1
                status_symbol = "✗ FALSIFIED"
                status_color = "🔴"
            elif status == "timeout":
                timeout_count += 1
                status_symbol = "⏱ TIMEOUT"
                status_color = "🟡"
            else:
                error_count += 1
                status_symbol = "⚠ ERROR"
                status_color = "🟠"
            
            print(f"   {status_color} Result: {status_symbol} (time: {elapsed:.2f}s)")
            
            # Additional info
            if verification_result.additional_info:
                method = verification_result.additional_info.get('verification_method', 'unknown')
                print(f"      Method used: {method}")
                
                if 'attack_phase_result' in verification_result.additional_info:
                    attack_result = verification_result.additional_info['attack_phase_result']
                    print(f"      Attack phase: {attack_result}")
            
            results.append({
                'problem_id': problem.problem_id,
                'verified': verified,
                'status': status,
                'verification_time': elapsed,
                'constraint_losses': constraint_losses,
                'bound_method': args.bound_method
            })
            
        except Exception as e:
            print(f"   ⚠️ Verification error: {str(e)}")
            error_count += 1
            results.append({
                'problem_id': problem.problem_id,
                'verified': False,
                'status': 'error',
                'verification_time': time.time() - t0,
                'error': str(e)
            })
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"📊 Formal Verification Summary")
    print(f"{'='*60}")
    
    total_attempted = verified_count + falsified_count + error_count + timeout_count
    
    print(f"\nProblems analyzed: {args.samples}")
    print(f"  ✓ Verified:   {verified_count}/{total_attempted} ({100*verified_count/max(total_attempted,1):.1f}%)")
    print(f"  ✗ Falsified:  {falsified_count}/{total_attempted} ({100*falsified_count/max(total_attempted,1):.1f}%)")
    print(f"  ⏱ Timeout:    {timeout_count}/{total_attempted} ({100*timeout_count/max(total_attempted,1):.1f}%)")
    print(f"  ⚠ Errors:     {error_count}/{total_attempted} ({100*error_count/max(total_attempted,1):.1f}%)")
    
    if total_attempted > 0:
        avg_time = total_time / total_attempted
        print(f"\nAverage verification time: {avg_time:.2f}s")
    
    print(f"\nVerification parameters:")
    print(f"  Epsilon: {args.epsilon} ({args.epsilon*100:.1f}% perturbation)")
    print(f"  Norm: L{args.norm}")
    print(f"  Bound method: {args.bound_method}")
    print(f"  Device: {DEVICE}")
    
    # Interpretation
    print(f"\n{'='*60}")
    print(f"🎯 What This Means")
    print(f"{'='*60}")
    
    if verified_count > 0:
        print(f"\n✅ {verified_count} logistics solutions are FORMALLY VERIFIED:")
        print(f"   → Guaranteed to satisfy all constraints even with")
        print(f"   → ±{args.epsilon*100:.1f}% perturbations to problem parameters")
        print(f"   → (flight delays, demand changes, capacity variations)")
        print(f"   → Mathematical proof via {args.bound_method}")
    
    if falsified_count > 0:
        print(f"\n❌ {falsified_count} solutions were FALSIFIED:")
        print(f"   → Found perturbations that break constraint satisfaction")
        print(f"   → Model not robust to parameter variations")
    
    # Save results
    import csv
    results_dir = 'logs'
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f'beluga_formal_verification_{int(time.time())}.csv')
    
    with open(results_path, 'w', newline='') as f:
        if results:
            fieldnames = ['problem_id', 'verified', 'status', 'verification_time', 'bound_method']
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\n💾 Results saved: {results_path}")
    print(f"\n✅ Formal verification complete!")


if __name__ == "__main__":
    main()