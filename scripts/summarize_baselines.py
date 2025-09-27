#!/usr/bin/env python3
"""
Summarize CPU and GPU baseline CSVs.

Reads files like:
  data/baselines/cpu/cpu_baselines_<ts>.csv
  data/baselines/gpu/gpu_baselines_<ts>.csv

Writes summaries into:
  data/baselines/cpu/summary/summary_cpu_baselines_<ts>.csv
  data/baselines/gpu/summary/summary_gpu_baselines_<ts>.csv
"""

import pathlib
import pandas as pd

BASE = pathlib.Path("data/baselines")
OUT_SUBDIR = "summary"


def pick_memory_column(df: pd.DataFrame):
    """Find best matching memory column name."""
    for name in ["memory_mb", "mem_mb", "memory_usage_mb", "memory_usage", "mem"]:
        if name in df.columns:
            return name
    return None


def pick_time_column(df: pd.DataFrame):
    """Find best matching time column name."""
    for name in ["verification_time", "time", "elapsed", "runtime_s"]:
        if name in df.columns:
            return name
    return None


def summarize_file(path: pathlib.Path):
    df = pd.read_csv(path)

    # Normalize verification column
    if "verified" in df.columns:
        df["verified"] = df["verified"].astype(bool)
    elif "status" in df.columns:
        df["verified"] = df["status"].astype(str).str.lower().eq("verified")
    else:
        df["verified"] = False

    # Normalize epsilon and norm
    if "epsilon" not in df.columns and "eps" in df.columns:
        df["epsilon"] = df["eps"]
    if "norm" not in df.columns and "p" in df.columns:
        df["norm"] = df["p"]

    # Memory column
    mem_col = pick_memory_column(df)
    if mem_col:
        df.rename(columns={mem_col: "memory_mb"}, inplace=True)
    else:
        df["memory_mb"] = pd.NA

    # Time column
    time_col = pick_time_column(df)
    if time_col:
        df.rename(columns={time_col: "verification_time"}, inplace=True)
    else:
        df["verification_time"] = pd.NA

    # Group summary
    group_cols = ["model", "norm", "epsilon"]
    ag = df.groupby(group_cols).agg(
        verification_rate=("verified", "mean"),
        runs=("verified", "size"),
        avg_time_s=("verification_time", "mean"),
        avg_mem_mb=("memory_mb", "mean"),
    ).reset_index()

    ag["avg_time_s"] = ag["avg_time_s"].fillna(0.0)
    ag["avg_mem_mb"] = ag["avg_mem_mb"].fillna(0.0)

    return ag


def process_dir(kind: str):
    src_dir = BASE / kind
    if not src_dir.exists():
        print(f"Skipping {kind}: no directory {src_dir}")
        return

    csvs = sorted(src_dir.glob(f"{kind}_baselines_*.csv"))
    if not csvs:
        print(f"No {kind} baseline CSVs found")
        return

    out_dir = src_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for csv in csvs:
        print("Read:", csv.name)
        try:
            summary = summarize_file(csv)
            out_path = out_dir / f"summary_{csv.name}"
            summary.to_csv(out_path, index=False)
            print("Wrote:", out_path)
        except Exception as e:
            print(f"ERROR summarizing {csv}: {e}")


if __name__ == "__main__":
    process_dir("cpu")
    process_dir("gpu")
    print("Done.")
