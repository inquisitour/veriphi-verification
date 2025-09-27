#!/usr/bin/env python3
"""
Summarize CPU baseline results across models, norms, and epsilons.

- Reads all `cpu_baselines_*.csv` in data/baselines/cpu
- Groups by model, norm, epsilon
- Computes verification_rate, avg_time, avg_mem
- Saves a new summary CSV with timestamp
"""

import os
import glob
import pandas as pd
from datetime import datetime

BASE_DIR = "data/baselines/cpu"

def main():
    files = glob.glob(os.path.join(BASE_DIR, "cpu_baselines_*.csv"))
    if not files:
        print("No baseline CSVs found in data/baselines/cpu")
        return

    print(f"Found {len(files)} baseline file(s).")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    summary = (
        df.groupby(["model", "norm", "epsilon"])
        .agg(
            verification_rate=("verified", "mean"),
            runs=("verified", "count"),
            avg_time_s=("time_s", "mean"),
            avg_mem_mb=("memory_usage_mb", "mean"),
        )
        .reset_index()
    )

    ts = int(datetime.now().timestamp())
    outdir = os.path.join(BASE_DIR, "summary")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(BASE_DIR, f"summary_cpu_baselines_{ts}.csv")
    summary.to_csv(out, index=False)

    print(f"\nRead:  {len(files)} file(s)")
    print(f"Wrote: {os.path.basename(out)}\n")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
