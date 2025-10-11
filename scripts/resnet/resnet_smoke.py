#!/usr/bin/env python3
"""
ResNet smoke test:
- ResNet-18 (CIFAR-10 style, 3x32x32 input)
- ResNet-50 (ImageNet style, 3x224x224 input)

If formal verification fails, falls back to FGSM/I-FGSM attack-only.

Note: ResNet stubs are made verification-friendly by swapping the first pooling
to AvgPool2d(2,2) (see src/core/models/resnet_stubs.py).
"""

import os
import torch
import traceback
from core import create_core_system
from core.models import create_resnet18, create_resnet50

DEVICE = os.environ.get("VERIPHI_DEVICE", "cpu")
EPS = 0.01
NORM = "inf"


def run_for_model(name: str, model, x, core):
    print(f"\n🚀 Verifying {name} (ε={EPS}, norm={NORM})")
    try:
        res = core.verify_robustness(model, x, epsilon=EPS, norm=NORM, timeout=30)
        print("Result:")
        print(f"  status:   {res.status.value}")
        print(f"  verified: {res.verified}")
        print(f"  time:     {res.verification_time:.3f}s")
        mem_mb = getattr(res, "memory_usage", None)
        if mem_mb is None and isinstance(res.additional_info, dict):
            mem_mb = res.additional_info.get("memory_usage_mb")
        print(f"  mem(MB):  {mem_mb}")
    except Exception as e:
        print("Verification failed with error:", str(e))
        print("⚠ Falling back to attack-only mode...")
        try:
            fgsm_res = core.attack_model(model, x, attack_name="fgsm", epsilon=EPS, norm=NORM)
            print("FGSM result:", fgsm_res.status)
            if hasattr(fgsm_res, "additional_info"):
                print("  info:", fgsm_res.additional_info)

            ifgsm_res = core.attack_model(model, x, attack_name="i-fgsm", epsilon=EPS, norm=NORM)
            print("I-FGSM result:", ifgsm_res.status)
            if hasattr(ifgsm_res, "additional_info"):
                print("  info:", ifgsm_res.additional_info)
        except Exception as e2:
            print("Attack fallback also failed:", e2)
            traceback.print_exc()


def main():
    core = create_core_system(use_attacks=True, device=DEVICE)

    # ResNet-18 (CIFAR-like, 32x32 RGB)
    try:
        r18 = create_resnet18(pretrained=False, num_classes=10)
        x18 = torch.randn(1, 3, 32, 32)
        run_for_model("ResNet-18 (CIFAR-10)", r18, x18, core)
    except Exception as e:
        print("ResNet-18 creation/run failed:", e)

    # ResNet-50 (ImageNet-like, 224x224 RGB)
    try:
        r50 = create_resnet50(pretrained=False, num_classes=1000)
        x50 = torch.randn(1, 3, 224, 224)
        run_for_model("ResNet-50 (ImageNet)", r50, x50, core)
    except Exception as e:
        print("ResNet-50 creation/run failed:", e)


if __name__ == "__main__":
    main()
