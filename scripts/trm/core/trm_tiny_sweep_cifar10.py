#!/usr/bin/env python3
"""
Batch verification sweep for TRM-MLP on CIFAR-10
Systematically tests multiple epsilon values and generates CSV results
"""

import os, time, csv, argparse
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from core import create_core_system
from core.models import create_trm_mlp

DEVICE = torch.device(os.environ.get("VERIPHI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--samples', type=int, default=100, help='Number of samples (default: 100)')
parser.add_argument('--checkpoint', type=str, help='Single checkpoint to test')
parser.add_argument('--eps', type=str, default='0.0078,0.031', help='Comma-separated epsilon values')
parser.add_argument('--timeout', type=int, default=120, help='Timeout per sample (default: 120s)')
parser.add_argument('--start-idx', type=int, default=0, help='Start sample index')
parser.add_argument('--end-idx', type=int, help='End sample index')
parser.add_argument('--bound', type=str, default='CROWN', 
                    choices=['CROWN', 'alpha-CROWN', 'beta-CROWN'], help='Bound method')
args = parser.parse_args()

NUM_SAMPLES = args.samples
EPSILONS = [float(e) for e in args.eps.split(',')]
BOUND_METHOD = args.bound

# Setup checkpoints
if args.checkpoint:
    checkpoint_name = os.path.basename(args.checkpoint).replace('.pt', '')
    CHECKPOINTS = {checkpoint_name: args.checkpoint}
else:
    CHECKPOINTS = {
        "Baseline": "checkpoints/trm_cifar10.pt",
        "IBP (eps=2/255)": "checkpoints/trm_cifar10_ibp_eps001.pt",
        "PGD (eps=8/255)": "checkpoints/trm_cifar10_adv_eps007.pt",
    }

def run_sweep(model_path, label, epsilons):
    tfm = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.CIFAR10(root="data/cifar10", train=False, download=True, transform=tfm)
    
    # Handle sample range
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx else NUM_SAMPLES
    actual_samples = end_idx - start_idx
    
    indices = torch.randint(0, len(test_ds), (NUM_SAMPLES,)).tolist()[start_idx:end_idx]
    
    print(f"\n{'='*60}")
    print(f"Model: {label}")
    print(f"Samples: {start_idx} to {end_idx} ({actual_samples} samples)")
    print(f"Bound method: {BOUND_METHOD}")
    print(f"{'='*60}")

    # Load model
    model = create_trm_mlp(
        x_dim=3*32*32, y_dim=10, z_dim=128, hidden=256,
        num_classes=10, H_cycles=2, L_cycles=2
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    core = create_core_system(use_attacks=True, device=DEVICE.type)
    results = []

    for eps in epsilons:
        v, f = 0, 0
        times, mems = [], []
        
        print(f"\n--- ε = {eps:.4f} ({eps*255:.1f}/255) ---")

        for idx in indices:
            x, y = test_ds[idx]
            x = x.view(1, -1).to(DEVICE)

            if DEVICE.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            t0 = time.time()
            res = core.verify_robustness(
                model, x, epsilon=eps, norm="inf", 
                timeout=args.timeout, bound_method=BOUND_METHOD
            )
            elapsed = time.time() - t0

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
                mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
            else:
                mem_mb = 0.0

            times.append(elapsed)
            mems.append(mem_mb)

            if res.status.value == "verified":
                v += 1
            elif res.status.value == "falsified":
                f += 1

        avg_time = sum(times) / len(times) if times else 0
        avg_mem = sum(mems) / len(mems) if mems else 0
        
        print(f"  Verified: {v}/{actual_samples} ({100*v/actual_samples:.1f}%)")
        print(f"  Falsified: {f}/{actual_samples}")
        print(f"  Avg time: {avg_time:.3f}s")
        print(f"  Avg GPU mem: {avg_mem:.1f} MB")

        results.append((eps, v, f, actual_samples, avg_time, avg_mem))

    return results

def plot_results(all_results):
    os.makedirs("plots", exist_ok=True)
    
    # Verified fraction vs epsilon
    plt.figure(figsize=(10, 6))
    for label, data in all_results.items():
        eps = [r[0] for r in data]
        verified_frac = [r[1]/r[3] for r in data]
        plt.plot(eps, verified_frac, marker='o', linewidth=2.5, markersize=8, label=label)
    
    plt.xlabel("ε (L∞ perturbation)", fontsize=13)
    plt.ylabel("Fraction Verified", fontsize=13)
    plt.title("Certified Robustness — TRM-MLP on CIFAR-10", fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("plots/trm_cifar10_verified_fraction.png", dpi=150)
    print("✅ Saved: plots/trm_cifar10_verified_fraction.png")

    # Verification time vs epsilon
    plt.figure(figsize=(8, 5))
    for label, data in all_results.items():
        eps = [r[0] for r in data]
        avg_time = [r[4] for r in data]
        plt.plot(eps, avg_time, marker='s', label=label)
    
    plt.xlabel("ε (L∞ perturbation)")
    plt.ylabel("Avg Verification Time (s)")
    plt.title("Verification Time vs Perturbation Size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/trm_cifar10_time_vs_eps.png", dpi=150)
    print("✅ Saved: plots/trm_cifar10_time_vs_eps.png")

    # Memory vs epsilon
    plt.figure(figsize=(8, 5))
    for label, data in all_results.items():
        eps = [r[0] for r in data]
        avg_mem = [r[5] for r in data]
        plt.plot(eps, avg_mem, marker='s', label=label)
    
    plt.xlabel("ε (L∞ perturbation)")
    plt.ylabel("Avg GPU Memory (MB)")
    plt.title("Verification Memory vs Perturbation Size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/trm_cifar10_memory_vs_eps.png", dpi=150)
    print("✅ Saved: plots/trm_cifar10_memory_vs_eps.png")

def main():
    os.makedirs("logs", exist_ok=True)
    all_results = {}

    for label, path in CHECKPOINTS.items():
        if os.path.exists(path):
            all_results[label] = run_sweep(path, label, EPSILONS)
        else:
            print(f"⚠️ Missing checkpoint: {path}")

    # Save CSV
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"logs/trm_cifar10_sweep_{BOUND_METHOD}_{timestamp}.csv"
    
    with open(csv_path, "w", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["model", "epsilon", "verified", "falsified", "total", 
                        "avg_time_s", "avg_mem_MB", "bound"])
        for label, data in all_results.items():
            for r in data:
                writer.writerow([label, *r, BOUND_METHOD])
    
    print(f"\n✅ Saved results → {csv_path}")

    # Generate plots
    plot_results(all_results)
    
    print("\n🎯 CIFAR-10 sweep complete!")

if __name__ == "__main__":
    main()