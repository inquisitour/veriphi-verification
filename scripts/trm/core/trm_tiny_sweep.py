#!/usr/bin/env python3
"""
TRM-MLP Robustness Sweep — Standard vs Adversarially Trained
Enhanced with bound method selection and sample range support
"""

import os
import csv
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from core import create_core_system
from core.models import create_trm_mlp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--samples', type=int, default=20, help='Number of samples to verify')
parser.add_argument('--eps', type=str, default=None, help='Comma-separated epsilon values')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to specific checkpoint')
parser.add_argument('--bound', type=str, default='CROWN', 
                   choices=['IBP', 'CROWN', 'alpha-CROWN', 'beta-CROWN'],
                   help='Bound method for verification')
parser.add_argument('--start-idx', type=int, default=0, help='Starting sample index (for distributed)')
parser.add_argument('--end-idx', type=int, default=None, help='Ending sample index (for distributed)')
parser.add_argument('--node-id', type=int, default=None, help='Node ID for filename (distributed runs)')
args = parser.parse_args()

# Configuration
DEVICE = torch.device(os.environ.get("VERIPHI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
torch.backends.cudnn.benchmark = True

if args.eps:
    EPSILONS = [float(e.strip()) for e in args.eps.split(',')]
else:
    EPSILONS = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1]

NUM_SAMPLES = args.samples
TIMEOUT = 900
BOUND_METHOD = args.bound

if args.checkpoint:
    checkpoint_name = os.path.basename(args.checkpoint).replace('.pt', '')
    CHECKPOINTS = {checkpoint_name: args.checkpoint}
else:
    CHECKPOINTS = {
        "Baseline": "checkpoints/trm_mnist.pt",
        "IBP (eps=1/255)": "checkpoints/trm_mnist_ibp_eps001_weights.pt",
        "PGD (eps=2/255)": "checkpoints/trm_mnist_adv_eps020.pt",
    }

def run_sweep(model_path, label, epsilons):
    tfm = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(root="data/mnist", train=False, download=True, transform=tfm)
    
    # Handle sample range for distributed execution
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx else NUM_SAMPLES
    actual_samples = end_idx - start_idx
    
    indices = torch.randint(0, len(test_ds), (NUM_SAMPLES,)).tolist()[start_idx:end_idx]
    
    print(f"Processing samples {start_idx} to {end_idx} ({actual_samples} samples)")
    print(f"Using bound method: {BOUND_METHOD}")

    model = create_trm_mlp(
        x_dim=28 * 28, y_dim=10, z_dim=128, hidden=256,
        num_classes=10, H_cycles=2, L_cycles=2
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    core = create_core_system(use_attacks=True, device=DEVICE)
    results = []

    for eps in epsilons:
        v, f = 0, 0
        times, mems = [], []
        print(f"\n=== [{label}] ε = {eps} | bound = {BOUND_METHOD} ===")

        for idx in indices:
            x, y = test_ds[idx]
            x = x.view(1, -1).to(DEVICE)

            if DEVICE.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            t0 = time.time()
            res = core.verify_robustness(
                model, x, epsilon=eps, norm="inf", 
                timeout=TIMEOUT, bound_method=BOUND_METHOD
            )
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0

            mem = None
            if DEVICE.type == "cuda":
                mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

            if res.verified:
                v += 1
            else:
                f += 1
            times.append(dt)
            if mem:
                mems.append(mem)

        avg_time = np.mean(times)
        avg_mem = np.mean(mems) if mems else 0.0
        results.append((eps, v, f, actual_samples, avg_time, avg_mem))
        print(f"{label}: ε={eps:.3f} → verified={v}/{actual_samples}, avg_time={avg_time:.3f}s, avg_mem={avg_mem:.1f}MB")

    return results

def plot_results(all_results):
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(8, 5))
    for label, data in all_results.items():
        eps = [r[0] for r in data]
        verified_frac = [r[1] / r[3] for r in data]
        plt.plot(eps, verified_frac, marker='o', label=f"{label}")
    plt.xlabel("ε (L∞ perturbation)")
    plt.ylabel("Fraction Verified")
    plt.title(f"Certified Robustness — TRM-MLP ({BOUND_METHOD})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/trm_robustness_{BOUND_METHOD}.png", dpi=150)
    print(f"✅ Saved: plots/trm_robustness_{BOUND_METHOD}.png")

    plt.figure(figsize=(8, 4))
    for label, data in all_results.items():
        eps = [r[0] for r in data]
        avg_time = [r[4] for r in data]
        plt.plot(eps, avg_time, marker='x', label=label)
    plt.xlabel("ε (L∞ perturbation)")
    plt.ylabel("Avg Verification Time (s)")
    plt.title("Verification Time vs Perturbation Size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/trm_time_vs_eps.png", dpi=150)
    print("✅ Saved: plots/trm_time_vs_eps.png")

    plt.figure(figsize=(8, 4))
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
    plt.savefig("plots/trm_memory_vs_eps.png", dpi=150)
    print("✅ Saved: plots/trm_memory_vs_eps.png")

def main():
    os.makedirs("logs", exist_ok=True)
    all_results = {}

    for label, path in CHECKPOINTS.items():
        if os.path.exists(path):
            all_results[label] = run_sweep(path, label, EPSILONS)
        else:
            print(f"⚠️ Missing checkpoint: {path}")

    # Save CSV with bound method and checkpoint name in filename
    checkpoint_base = os.path.basename(args.checkpoint).replace('.pt', '') if args.checkpoint else "default"
    csv_path = f"logs/trm_sweep_{checkpoint_base}_{BOUND_METHOD}_{args.start_idx}_{args.end_idx or NUM_SAMPLES}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "epsilon", "verified", "falsified", "total", "avg_time_s", "avg_mem_MB", "bound"])
        for label, data in all_results.items():
            for r in data:
                writer.writerow([label, *r, BOUND_METHOD])
    print(f"\n✅ Saved results → {csv_path}")

    plot_results(all_results)
    print("\n🎯 Sweep complete!")

if __name__ == "__main__":
    main()