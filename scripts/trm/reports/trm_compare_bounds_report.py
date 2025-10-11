#!/usr/bin/env python3
"""
Compare multiple TRM sweeps (CROWN, α, β) → PDF report.
"""

import os, glob, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet

os.makedirs("reports", exist_ok=True)

# find all bound sweeps
csvs = sorted(glob.glob("runs/trm_sweep_*.csv") + glob.glob("logs/trm_sweep_*.csv"))
if not csvs:
    raise FileNotFoundError("No sweep CSVs found in runs/ or logs/.")

dfs = []
for f in csvs:
    df = pd.read_csv(f)
    if "alpha" in f: df["bound"]="α-CROWN"
    elif "beta" in f: df["bound"]="β-CROWN"
    elif "crown" in f: df["bound"]="CROWN"
    else: df["bound"]="Other"
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
if "verified_fraction" not in df:
    df["verified_fraction"]=df["verified"]/df["total"]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# === Plots ===
plt.figure(figsize=(6,4))
for b,m in zip(["CROWN","α-CROWN","β-CROWN"],["o","s","^"]):
    d=df[df["bound"]==b]
    if not d.empty: plt.plot(d["epsilon"], d["verified_fraction"], marker=m, label=b)
plt.xlabel("ε"); plt.ylabel("Verified Fraction")
plt.title("TRM Robustness Across Bounds")
plt.grid(True, alpha=0.3); plt.legend()
plot_path = f"reports/compare_bounds_verified_{timestamp}.png"
plt.savefig(plot_path,dpi=200); plt.close()

# === PDF ===
# safer summary generation (handles varying columns)
cols = [c for c in ["verified","falsified","total"] if c in df.columns]
summary = df.groupby("bound")[cols].sum().reset_index()

if "total" not in summary:
    summary["total"] = summary["verified"].sum()
summary["verified_fraction"] = (summary["verified"] / summary["total"]).round(2)

styles=getSampleStyleSheet(); story=[]
story.append(Paragraph("<b>TRM Bound Comparison</b>",styles["Title"]))
story.append(Spacer(1,12))
story.append(Image(plot_path,width=400,height=280))
story.append(Spacer(1,12))

data=[summary.columns.tolist()]+summary.values.tolist()
t=Table(data)
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),colors.grey),
    ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
    ('ALIGN',(0,0),(-1,-1),'CENTER'),
    ('GRID',(0,0),(-1,-1),0.5,colors.black)
]))
story.append(t)

out_pdf=f"reports/trm_compare_bounds_report_{timestamp}.pdf"
SimpleDocTemplate(out_pdf,pagesize=A4).build(story)
print(f"✅ Saved {out_pdf}")
