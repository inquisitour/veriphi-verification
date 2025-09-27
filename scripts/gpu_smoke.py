#!/usr/bin/env python3
import torch, time, os
from core import create_core_system
from core.models import create_test_model, create_sample_input

def main():
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"CUDA available: {use_cuda}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Selected device: {device}")

    core = create_core_system(use_attacks=True, device=device)
    model = create_test_model("tiny")
    x = create_sample_input("tiny")

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.time()
    res = core.verify_robustness(model, x, epsilon=0.1, norm="inf", timeout=30)
    if use_cuda:
        torch.cuda.synchronize()
    dt = time.time() - t0

    peak_mb = None
    if use_cuda:
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print("\nResult:")
    print(f"  status:   {res.status.value}")
    print(f"  verified: {res.verified}")
    print(f"  time:     {dt:.3f}s")
    print(f"  mem(MB):  {peak_mb if peak_mb is not None else res.additional_info.get('memory_usage_mb')}")

if __name__ == "__main__":
    # PYTHONPATH must include ./src when running from repo root
    if "PYTHONPATH" not in os.environ or "src" not in os.environ["PYTHONPATH"]:
        os.environ["PYTHONPATH"] = f"{os.getcwd()}/src:" + os.environ.get("PYTHONPATH", "")
    main()
