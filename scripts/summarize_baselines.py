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
import time

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
    for name in ["verification_time", "time_s", "time", "elapsed", "runtime_s"]:
        if name in df.columns:
            return name
    return None


def summarize_file(path: pathlib.Path):
    """Summarize a baseline CSV file."""
    print(f"Processing: {path.name}")
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
        df["memory_mb"] = df[mem_col]
    else:
        df["memory_mb"] = pd.NA

    # Time column
    time_col = pick_time_column(df)
    if time_col:
        df["verification_time"] = df[time_col]
    else:
        df["verification_time"] = pd.NA

    # Group and aggregate
    group_cols = ["model", "norm", "epsilon"]
    
    # Only group by columns that exist
    existing_group_cols = [col for col in group_cols if col in df.columns]
    
    if not existing_group_cols:
        print(f"Warning: No grouping columns found in {path.name}")
        return None

    try:
        ag = df.groupby(existing_group_cols).agg(
            verification_rate=("verified", "mean"),
            runs=("verified", "size"),
            avg_time_s=("verification_time", "mean"),
            avg_mem_mb=("memory_mb", "mean"),
        ).reset_index()

        # Fill NaN values with reasonable defaults
        ag["avg_time_s"] = ag["avg_time_s"].fillna(0.0)
        ag["avg_mem_mb"] = ag["avg_mem_mb"].fillna(0.0)
        ag["verification_rate"] = ag["verification_rate"].fillna(0.0)

        print(f"  → Created summary with {len(ag)} rows")
        return ag
        
    except Exception as e:
        print(f"Error processing {path.name}: {e}")
        return None


def process_dir(kind: str):
    """Process all baseline files in a directory."""
    src_dir = BASE / kind
    if not src_dir.exists():
        print(f"Skipping {kind}: no directory {src_dir}")
        return

    # Find baseline CSV files
    csvs = sorted(src_dir.glob(f"{kind}_baselines_*.csv"))
    if not csvs:
        print(f"No {kind} baseline CSVs found in {src_dir}")
        return

    out_dir = src_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in csvs:
        try:
            summary = summarize_file(csv_path)
            if summary is not None:
                # Create output filename
                timestamp = int(time.time())
                out_filename = f"summary_{csv_path.stem}_{timestamp}.csv"
                out_path = out_dir / out_filename
                
                # Save summary
                summary.to_csv(out_path, index=False)
                print(f"  → Saved: {out_path}")
            else:
                print(f"  → Skipped: {csv_path.name} (no data to summarize)")
                
        except Exception as e:
            print(f"ERROR summarizing {csv_path}: {e}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("BASELINE SUMMARIZATION TOOL")
    print("=" * 60)
    
    process_dir("cpu")
    process_dir("gpu")
    
    print("\n" + "=" * 60)
    print("SUMMARY COMPLETE")
    print("=" * 60)
    
    # Show what was created
    for kind in ["cpu", "gpu"]:
        summary_dir = BASE / kind / OUT_SUBDIR
        if summary_dir.exists():
            summary_files = list(summary_dir.glob("*.csv"))
            print(f"{kind.upper()}: {len(summary_files)} summary files in {summary_dir}")


if __name__ == "__main__":
    main()