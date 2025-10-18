#!/usr/bin/env python3
"""
Single-sample verification for TRM-MLP on CIFAR-10
Tests 10 random samples to verify robustness
"""

import os, random, argparse
import torch
from torchvision import datasets, transforms

from core import create_core_system
from core.models import create_trm_mlp

DEVICE = torch.device(os.environ.get("VERIPHI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eps', type=float, default=0.0078, help='Epsilon (default: 0.0078 = 2/255)')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to verify')
    parser.add_argument('--timeout', type=int, default=120, help='Timeout per sample (seconds)')
    args = parser.parse_args()

    # Load model
    model = create_trm_mlp(
        x_dim=3*32*32, y_dim=10, z_dim=128, hidden=256,
        num_classes=10, H_cycles=2, L_cycles=2
    ).to(DEVICE)
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()
    print(f"✅ Loaded: {args.checkpoint}")

    # Create verification system
    core = create_core_system(use_attacks=True, device=DEVICE.type)

    # Load CIFAR-10 test set
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root="data/cifar10", train=False, download=True, transform=tfm)

    # Random samples
    idxs = random.sample(range(len(ds)), args.samples)
    
    print(f"\n🔍 Verifying {args.samples} samples @ ε={args.eps:.4f} ({args.eps*255:.1f}/255)")
    print(f"   Timeout: {args.timeout}s per sample")
    print(f"   Device: {DEVICE}\n")

    ok = fail = err = 0
    for i in idxs:
        x, y = ds[i]
        x = x.view(1, -1).to(DEVICE)  # flatten to [1, 3072]
        
        res = core.verify_robustness(model, x, epsilon=args.eps, norm="inf", timeout=args.timeout)
        
        status = "✓ VERIFIED" if res.verified else "✗ FALSIFIED" if res.status.value == "falsified" else "⚠ OTHER"
        print(f"Sample {i:<5}  {status:<15}  time={res.verification_time:.3f}s")
        
        if res.status.value == "verified": ok += 1
        elif res.status.value == "falsified": fail += 1
        else: err += 1

    print(f"\n{'='*50}")
    print(f"Summary @ ε={args.eps:.4f} ({args.eps*255:.1f}/255):")
    print(f"  ✓ Verified:   {ok}/{args.samples} ({100*ok/args.samples:.1f}%)")
    print(f"  ✗ Falsified:  {fail}/{args.samples} ({100*fail/args.samples:.1f}%)")
    print(f"  ⚠ Other:      {err}/{args.samples}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()