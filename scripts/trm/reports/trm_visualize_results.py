#!/usr/bin/env python3
"""
Visualize TRM robustness sweep results.
Generates plots from CSV data: model,epsilon,verified,falsified,total,avg_time_s,avg_mem_MB
Output: Multiple PNG plots in plots/ and reports/
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Setup
os.makedirs("plots", exist_ok=True)
os.makedirs("reports", exist_ok=True)
sns.set_style("whitegrid")

# Load CSV data
csvs = sorted(glob.glob("logs/trm_mnist_sweep_*.csv"))
if not csvs:
    print("❌ No sweep CSVs found in logs/")
    exit(1)

df = pd.read_csv(csvs[-1])
print(f"📄 Loaded: {csvs[-1]}")
print(f"   Rows: {len(df)}, Columns: {list(df.columns)}")

# Add verified_fraction if missing
if "verified_fraction" not in df.columns:
    df["verified_fraction"] = df["verified"] / df["total"]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ========================================
# Plot 1: Verified Fraction vs Epsilon (Main Result)
# ========================================
plt.figure(figsize=(10, 6))
for model in df["model"].unique():
    data = df[df["model"] == model]
    plt.plot(data["epsilon"], data["verified_fraction"], 
             marker='o', linewidth=2.5, markersize=8, label=model)

plt.xlabel("ε (L∞ perturbation)", fontsize=13)
plt.ylabel("Fraction Verified", fontsize=13)
plt.title("Certified Robustness — TRM-MLP on MNIST", fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11, loc='best')
plt.tight_layout()
plot1 = f"plots/trm_results_{timestamp}.png"
plt.savefig(plot1, dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ Saved plot: {plot1}")

# ========================================
# Plot 2: Verification Time vs Epsilon
# ========================================
plt.figure(figsize=(10, 6))
for model in df["model"].unique():
    data = df[df["model"] == model]
    plt.plot(data["epsilon"], data["avg_time_s"], 
             marker='s', linewidth=2.5, markersize=8, label=model)

plt.xlabel("ε (L∞ perturbation)", fontsize=13)
plt.ylabel("Avg Verification Time (s)", fontsize=13)
plt.title("Verification Time vs Perturbation Size", fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11, loc='best')
plt.tight_layout()
plot2 = f"plots/trm_results_{timestamp}.png".replace(".png", "_time.png")
plt.savefig(plot2, dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ Saved plot: {plot2}")

# ========================================
# Plot 3: GPU Memory Usage
# ========================================
plt.figure(figsize=(10, 6))
for model in df["model"].unique():
    data = df[df["model"] == model]
    plt.plot(data["epsilon"], data["avg_mem_MB"], 
             marker='^', linewidth=2.5, markersize=8, label=model)

plt.xlabel("ε (L∞ perturbation)", fontsize=13)
plt.ylabel("GPU Memory (MB)", fontsize=13)
plt.title("GPU Memory Usage During Verification", fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11, loc='best')
plt.tight_layout()
plot3 = f"plots/trm_results_{timestamp}.png".replace(".png", "_memory.png")
plt.savefig(plot3, dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ Saved plot: {plot3}")

# ========================================
# Plot 4: Heatmap - Verified Fraction (if multiple models/bounds)
# ========================================
# Check if we have data suitable for heatmap
if "model" in df.columns and len(df["model"].unique()) > 1:
    pivot_data = df.pivot_table(
        values='verified_fraction', 
        index='epsilon', 
        columns='model', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_data.T, annot=True, fmt='.2f', cmap='RdYlGn', 
                cbar_kws={'label': 'Verified Fraction'}, vmin=0, vmax=1)
    plt.title("Verified Fraction Heatmap", fontsize=15, fontweight='bold')
    plt.xlabel("ε (L∞ perturbation)", fontsize=13)
    plt.ylabel("Model", fontsize=13)
    plt.tight_layout()
    heatmap_path = "reports/heatmap_verified_fraction.png"
    plt.savefig(heatmap_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved heatmap: {heatmap_path}")
else:
    print("⚠️ Single model detected - skipping heatmap")

# ========================================
# Plot 5: Verified Fraction Curve (for reports/)
# ========================================
plt.figure(figsize=(10, 6))
for model in df["model"].unique():
    data = df[df["model"] == model]
    plt.plot(data["epsilon"], data["verified_fraction"], 
             marker='o', linewidth=3, markersize=10, label=model, alpha=0.8)

plt.xlabel("ε (L∞ perturbation)", fontsize=14, fontweight='bold')
plt.ylabel("Verified Fraction", fontsize=14, fontweight='bold')
plt.title("TRM Certified Robustness", fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.4, linestyle='--')
plt.legend(fontsize=12, loc='best', framealpha=0.9)
plt.tight_layout()
curve_path = "reports/verified_fraction_curve.png"
plt.savefig(curve_path, dpi=250, bbox_inches='tight')
plt.close()
print(f"✅ Saved curve: {curve_path}")

# ========================================
# Plot 6: Bar Chart - Total Verified/Falsified by Model
# ========================================
summary = df.groupby("model").agg({
    "verified": "sum",
    "falsified": "sum"
}).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(summary))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], summary["verified"], 
               width, label='Verified', color='green', alpha=0.7)
bars2 = ax.bar([i + width/2 for i in x], summary["falsified"], 
               width, label='Falsified', color='red', alpha=0.7)

ax.set_xlabel('Model', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.set_title('Total Verified vs Falsified Samples', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(summary["model"])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
bar_path = f"plots/trm_results_{timestamp}_barchart.png"
plt.savefig(bar_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ Saved bar chart: {bar_path}")

# ========================================
# Plot 7: Confidence Drop Histogram (if available)
# ========================================
if "confidence_drop" in df.columns:
    plt.figure(figsize=(10, 6))
    for model in df["model"].unique():
        data = df[df["model"] == model]
        plt.hist(data["confidence_drop"], bins=20, alpha=0.6, label=model)
    
    plt.xlabel("Confidence Drop", fontsize=13)
    plt.ylabel("Frequency", fontsize=13)
    plt.title("Attack Confidence Drop Distribution", fontsize=15, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    hist_path = "reports/attack_confidence_hist.png"
    plt.savefig(hist_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved histogram: {hist_path}")
else:
    print("⚠️ No 'confidence_drop' column found in logs. Skipping histogram.")

# ========================================
# Summary Statistics
# ========================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

for model in df["model"].unique():
    model_data = df[df["model"] == model]
    print(f"\n📊 {model}")
    print(f"   Total samples: {model_data['total'].sum()}")
    print(f"   Total verified: {model_data['verified'].sum()}")
    print(f"   Total falsified: {model_data['falsified'].sum()}")
    print(f"   Avg verified fraction: {model_data['verified_fraction'].mean()*100:.1f}%")
    print(f"   Avg verification time: {model_data['avg_time_s'].mean():.3f}s")
    print(f"   Avg GPU memory: {model_data['avg_mem_MB'].mean():.1f} MB")
    
    # Best ε for this model
    best_row = model_data.loc[model_data['verified_fraction'].idxmax()]
    print(f"   Best ε: {best_row['epsilon']} ({best_row['verified_fraction']*100:.1f}% verified)")

print("\n" + "="*60)
print("✅ Visualization complete!")
print(f"   Plots saved to: plots/")
print(f"   Reports saved to: reports/")
print("="*60)