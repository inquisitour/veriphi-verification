#!/usr/bin/env python3
import os, time, csv, json, random
from datetime import datetime
import torch

from core import create_core_system
from core.models import create_test_model, create_sample_input

def run():
    # ensure PYTHONPATH works when run from repo root
    if "PYTHONPATH" not in os.environ or "src" not in os.environ["PYTHONPATH"]:
        os.environ["PYTHONPATH"] = f"{os.getcwd()}/src:" + os.environ.get("PYTHONPATH", "")

    torch.manual_seed(0); random.seed(0)

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    models   = ["tiny","linear","conv"]
    norms    = ["inf","2"]
    epsilons = [0.01, 0.05, 0.1, 0.2]

    out_dir = "data/baselines/gpu" if use_cuda else "data/baselines/cpu"
    os.makedirs(out_dir, exist_ok=True)

    ts = int(time.time())
    csv_path = os.path.join(out_dir, f"{'gpu' if use_cuda else 'cpu'}_baselines_{ts}.csv")

    core = create_core_system(use_attacks=True, device=device)
    rows = []

    for mname in models:
        model = create_test_model(mname)
        x = create_sample_input(mname)

        for norm in norms:
            for eps in epsilons:
                if use_cuda:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                t0 = time.time()
                res = core.verify_robustness(model, x, epsilon=eps, norm="inf" if norm == "inf" else "2", timeout=30)
                if use_cuda:
                    torch.cuda.synchronize()
                dt = time.time() - t0

                peak_mb = None
                if use_cuda:
                    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

                mem = peak_mb if peak_mb is not None else res.additional_info.get("memory_usage_mb")

                rows.append({
                    "model": mname,
                    "norm": norm,
                    "epsilon": eps,
                    "status": res.status.value,
                    "verified": bool(res.verified),
                    "time_s": float(dt),
                    "memory_mb": float(mem) if mem is not None else None,
                    "method": res.additional_info.get("method", "attack-guided")
                })
                print(f"{mname:6s} | norm={norm:>3s} | eps={eps:<4} → {res.status.value:9s} | {dt:.3f}s")

    # write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows → {csv_path}")

if __name__ == "__main__":
    run()
