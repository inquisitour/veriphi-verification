#!/usr/bin/env python3
"""
TRM-MLP Robustness Sweep — Standard vs Adversarially Trained
=============================================================

Runs attack-guided verification for multiple ε (L∞) values and logs:
 - verified / falsified counts
 - average verification time
 - average GPU memory usage
 - comparison between standard and adversarial TRM-MLP
Generates plots for robustness, time, and memory trends.
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
args = parser.parse_args()

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------
DEVICE = torch.device(os.environ.get("VERIPHI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
torch.backends.cudnn.benchmark = True
EPSILONS = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1]
NUM_SAMPLES = args.samples
TIMEOUT = 30  # seconds
CHECKPOINTS = {
    "Standard TRM": "checkpoints/trm_mnist.pt",
    "Adversarial TRM": "checkpoints/trm_mnist_adv.pt",
}


# ----------------------------------------------------------
# Sweep runner
# ----------------------------------------------------------
def run_sweep(model_path, label, epsilons):
    tfm = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(root="data/mnist", train=False, download=True, transform=tfm)
    indices = torch.randint(0, len(test_ds), (NUM_SAMPLES,)).tolist()

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
        print(f"\n=== [{label}] ε = {eps} ===")

        for idx in indices:
            x, y = test_ds[idx]
            x = x.view(1, -1).to(DEVICE)

            if DEVICE.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            t0 = time.time()
            res = core.verify_robustness(model, x, epsilon=eps, norm="inf", timeout=TIMEOUT)
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
        results.append((eps, v, f, NUM_SAMPLES, avg_time, avg_mem))
        print(f"{label}: ε={eps:.3f} → verified={v}/{NUM_SAMPLES}, avg_time={avg_time:.3f}s, avg_mem={avg_mem:.1f}MB")

    return results


# ----------------------------------------------------------
# Plotting utilities
# ----------------------------------------------------------
def plot_results(all_results):
    os.makedirs("plots", exist_ok=True)

    # Robustness Curve
    plt.figure(figsize=(8, 5))
    for label, data in all_results.items():
        eps = [r[0] for r in data]
        verified_frac = [r[1] / r[3] for r in data]
        plt.plot(eps, verified_frac, marker='o', label=f"{label}")
    plt.xlabel("ε (L∞ perturbation)")
    plt.ylabel("Fraction Verified")
    plt.title("Certified Robustness — TRM-MLP on MNIST")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/trm_robustness_comparison.png", dpi=150)
    print("✅ Saved: plots/trm_robustness_comparison.png")

    # Time vs ε
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

    # Memory vs ε
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


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    os.makedirs("logs", exist_ok=True)
    all_results = {}

    for label, path in CHECKPOINTS.items():
        if os.path.exists(path):
            all_results[label] = run_sweep(path, label, EPSILONS)
        else:
            print(f"⚠️ Missing checkpoint: {path}")

    # Save combined CSV
    csv_path = "logs/trm_robustness_sweep_full.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "epsilon", "verified", "falsified", "total", "avg_time_s", "avg_mem_MB"])
        for label, data in all_results.items():
            for r in data:
                writer.writerow([label, *r])
    print(f"\n✅ Saved results → {csv_path}")

    # Plot
    plot_results(all_results)
    print("\n🎯 Sweep complete — results logged and plots generated!")


if __name__ == "__main__":
    main()
