#!/usr/bin/env python3
import os, random
import torch
from torchvision import datasets, transforms

from core import create_core_system
from core.models import create_trm_mlp

DEVICE = torch.device(os.environ.get("VERIPHI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

def main():
    # Build the exact same model skeleton and load weights
    model = create_trm_mlp(
        x_dim=28*28, y_dim=10, z_dim=128, hidden=256,
        num_classes=10, H_cycles=2, L_cycles=2
    ).to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/trm_mnist_adv.pt", map_location=DEVICE))
    model.eval()

    core = create_core_system(use_attacks=True, device=DEVICE.type)

    # Load MNIST test set
    ds = datasets.MNIST(root="data/mnist", train=False, download=True,
                        transform=transforms.ToTensor())

    # pick 10 random test images and verify local robustness
    idxs = random.sample(range(len(ds)), 10)
    eps   = 0.03   # L_inf on [0,1] pixels
    norm  = "inf" # or "2"

    ok = fail = err = 0
    for i in idxs:
        x, y = ds[i]
        x = x.view(1, -1).to(DEVICE)  # flatten
        res = core.verify_robustness(model, x, epsilon=eps, norm=norm, timeout=30)
        print(f"idx={i:<5}  status={res.status.value:<9}  verified={res.verified}")
        if res.status.value == "verified": ok += 1
        elif res.status.value == "falsified": fail += 1
        else: err += 1

    print(f"\nSummary @ ε={eps}, L_{norm}:  verified={ok}, falsified={fail}, other={err}")

if __name__ == "__main__":
    main()
