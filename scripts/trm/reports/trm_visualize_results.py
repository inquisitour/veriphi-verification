#!/usr/bin/env python3
"""
Visualize TRM verification and attack-guided results:
 - Verified vs falsified heatmap by sample index
 - Attack confidence-drop histogram
 - ε (epsilon) vs verified fraction overlay
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Directories
LOG_DIR = "logs"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# --- Load all available sweep files ---
csvs = [f for f in os.listdir(LOG_DIR) if f.startswith("trm_robustness_sweep") and f.endswith(".csv")]
if not csvs:
    raise FileNotFoundError("No sweep logs found in 'logs/'. Please run trm_tiny_sweep.py first.")

dfs = []
for c in csvs:
    df = pd.read_csv(os.path.join(LOG_DIR, c))
    # Infer bound method if not present
    if "bound" not in df.columns:
        if "alpha" in c:
            df["bound"] = "α-CROWN"
        elif "beta" in c:
            df["bound"] = "β-CROWN"
        else:
            df["bound"] = "CROWN"
    # Compute verified fraction if missing
    if "verified_fraction" not in df.columns:
        if "verified" in df.columns and "total" in df.columns:
            df["verified_fraction"] = df["verified"] / df["total"]
        else:
            print(f"⚠️ Skipping {c}: missing verified/total columns")
            continue
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# --- 1️⃣ Verified vs falsified heatmap ---
pivot = df.pivot_table(index="epsilon", columns="bound", values="verified_fraction", aggfunc="mean")
plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".2f")
plt.title("Verified Fraction Heatmap (ε vs Bound Method)")
plt.xlabel("Bound Method")
plt.ylabel("ε (Perturbation Radius)")
plt.tight_layout()
plot1 = os.path.join(REPORT_DIR, "heatmap_verified_fraction.png")
# ensure folder exists
os.makedirs("plots", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"plots/trm_results_{timestamp}.png"
plt.savefig(out_path, bbox_inches="tight")
print(f"✅ Saved plot: {out_path}")

plt.close()

# --- 2️⃣ Attack confidence drop histogram ---
if "confidence_drop" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="confidence_drop", hue="bound", bins=30, kde=True)
    plt.title("Distribution of Attack Confidence Drops")
    plt.xlabel("Confidence Drop (original - attacked)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plot2 = os.path.join(REPORT_DIR, "attack_confidence_hist.png")
    # ensure folder exists
    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"plots/trm_results_{timestamp}.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"✅ Saved plot: {out_path}")
    plt.close()
else:
    print("⚠️ No 'confidence_drop' column found in logs. Skipping histogram.")

# --- 3️⃣ Epsilon vs Verified Fraction Overlay ---
plt.figure(figsize=(8,5))
for method in df["bound"].unique():
    subset = df[df["bound"] == method]
    plt.plot(subset["epsilon"], subset["verified_fraction"], marker="o", label=method)
plt.title("Verified Fraction vs ε")
plt.xlabel("ε (L∞ perturbation)")
plt.ylabel("Verified Fraction")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot3 = os.path.join(REPORT_DIR, "verified_fraction_curve.png")
# ensure folder exists
os.makedirs("plots", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"plots/trm_results_{timestamp}.png"
plt.savefig(out_path, bbox_inches="tight")
print(f"✅ Saved plot: {out_path}")

plt.close()

print(f"✅ Saved visualizations:")
print(f" - {plot1}")
if os.path.exists(os.path.join(REPORT_DIR, 'attack_confidence_hist.png')):
    print(f" - {os.path.join(REPORT_DIR, 'attack_confidence_hist.png')}")
print(f" - {plot3}")
