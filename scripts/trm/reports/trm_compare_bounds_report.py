#!/usr/bin/env python3
"""
Compare multiple TRM sweeps (different bounds or models) → PDF report.
Works with CSV format: model,epsilon,verified,falsified,total,avg_time_s,avg_mem_MB
"""

import os, glob, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet

os.makedirs("reports", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Find all sweep CSVs
csvs = sorted(glob.glob("runs/trm_sweep_*.csv") + glob.glob("logs/trm_robustness_sweep*.csv"))
if not csvs:
    print("❌ No sweep CSVs found")
    exit(1)

# Load and combine all CSVs
dfs = []
for f in csvs:
    df = pd.read_csv(f)
    
    # Try to infer bound from filename if 'bound' column missing
    if "bound" not in df.columns:
        if "alpha" in f.lower():
            df["bound"] = "α-CROWN"
        elif "beta" in f.lower():
            df["bound"] = "β-CROWN"
        elif "crown" in f.lower():
            df["bound"] = "CROWN"
        else:
            # If no bound info, use model name
            df["bound"] = df.get("model", "Unknown")
    
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Add verified_fraction if missing
if "verified_fraction" not in df.columns:
    df["verified_fraction"] = df["verified"] / df["total"]

print(f"📄 Loaded {len(csvs)} CSV files")
print(f"   Unique bounds/models: {df['bound'].unique().tolist()}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# === Generate Comparison Plot ===
plt.figure(figsize=(8, 5))
markers = ['o', 's', '^', 'd', 'v', '<', '>']
for idx, bound in enumerate(df["bound"].unique()):
    data = df[df["bound"] == bound]
    marker = markers[idx % len(markers)]
    plt.plot(data["epsilon"], data["verified_fraction"], 
             marker=marker, linewidth=2, label=bound)

plt.xlabel("ε (L∞ perturbation)", fontsize=12)
plt.ylabel("Verified Fraction", fontsize=12)
plt.title("TRM Robustness Comparison", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plot_path = f"plots/compare_bounds_{timestamp}.png"
plt.savefig(plot_path, dpi=200)
plt.close()
print(f"✅ Saved plot: {plot_path}")

# === Generate PDF ===
pdf_path = f"reports/trm_compare_bounds_report_{timestamp}.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=A4)
styles = getSampleStyleSheet()
story = []

# Title
story.append(Paragraph("<b>TRM Bound/Model Comparison Report</b>", styles["Title"]))
story.append(Spacer(1, 12))
story.append(Paragraph(
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
    f"Comparing: {', '.join(df['bound'].unique())}",
    styles["Normal"]
))
story.append(Spacer(1, 20))

# Plot
story.append(Paragraph("<b>Verification Comparison</b>", styles["Heading2"]))
story.append(Spacer(1, 12))
story.append(Image(plot_path, width=480, height=300))
story.append(Spacer(1, 20))

# Summary Table
story.append(Paragraph("<b>Average Verified Fractions by Bound/Model</b>", styles["Heading2"]))
summary = df.groupby("bound").agg({
    "verified": "sum",
    "falsified": "sum", 
    "total": "sum",
    "verified_fraction": "mean"
}).reset_index()
summary["verified_fraction"] = summary["verified_fraction"].apply(lambda x: f"{x*100:.1f}%")

table_data = [["Bound/Model", "Total Verified", "Total Falsified", "Total Samples", "Avg Ver. Fraction"]]
table_data += summary.values.tolist()

t = Table(table_data)
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
]))
story.append(t)

doc.build(story)
print(f"✅ PDF saved: {pdf_path}")