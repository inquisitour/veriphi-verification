#!/usr/bin/env python3
import os, torch
from core import create_core_system
from core.models import create_test_model

DEVICE = os.environ.get("VERIPHI_DEVICE", "cuda")

def main():
    core = create_core_system(use_attacks=True, device=DEVICE)
    model = create_test_model("trm-mlp").to(DEVICE).eval()

    # synthetic “puzzle embedding” input
    x = torch.randn(1, 512, device=DEVICE)

    res = core.verify_robustness(model, x, epsilon=0.01, norm="inf", timeout=30)
    print("\nResult:")
    print("  status:", res.status.value)
    print("  verified:", res.verified)
    print("  time:", f"{res.verification_time:.3f}s")

if __name__ == "__main__":
    main()
