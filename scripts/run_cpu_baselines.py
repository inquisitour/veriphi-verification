#!/usr/bin/env python3
"""
Run CPU baselines for Tiny/Linear/Conv models across epsilon values and norms.
Saves results into data/baselines/cpu with timestamped CSVs.
"""

import os, time, json, csv, random
from datetime import datetime
import torch
from core import create_core_system
from core.models import create_test_model, create_sample_input

# Reproducibility
torch.manual_seed(0)
random.seed(0)

def main():
    models   = ["tiny", "linear", "conv"]
    norms    = ["inf", "2"]
    epsilons = [0.01, 0.05, 0.1, 0.2]

    core = create_core_system(use_attacks=True, device="cpu")
    rows = []

    for mname in models:
        model = create_test_model(mname)
        x     = create_sample_input(mname)

        for norm in norms:
            for eps in epsilons:
                print(f"\n>>> Running {mname} | norm={norm} | eps={eps}")
                res = core.verify_robustness(model, x, epsilon=eps, norm=norm, timeout=30)

                row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "model": mname,
                    "norm": norm,
                    "epsilon": eps,
                    "status": res.status.value,
                    "verified": res.verified,
                    "time_s": res.verification_time,
                    "memory_usage_mb": res.memory_usage,
                }
                rows.append(row)

                print(f"{mname:6s} | norm={norm:3s} | eps={eps:.3f} "
                      f"→ {res.status.value:9s} | {res.verification_time:.3f}s")

    # Save results
    os.makedirs("data/baselines", exist_ok=True)
    ts = int(time.time())
    fname = f"data/baselines/cpu/cpu_baselines_{ts}.csv"
    with open(fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows → {fname}")

if __name__ == "__main__":
    main()
