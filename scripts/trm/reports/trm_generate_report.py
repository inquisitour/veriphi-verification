#!/usr/bin/env python3
"""
Generate a PDF report summarizing TRM robustness results from latest sweeps.
"""

import os, glob, datetime, pandas as pd, matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet

os.makedirs("reports", exist_ok=True)
os.makedirs("plots", exist_ok=True)

candidates = sorted(glob.glob("runs/trm_sweep_*.csv") + glob.glob("logs/trm_sweep_*.csv"))
if not candidates:
    raise FileNotFoundError("No sweep CSVs found in runs/ or logs/.")

# Merge all runs into one DataFrame
df = pd.concat([pd.read_csv(f) for f in candidates], ignore_index=True)
csv_path = candidates[-1]
print(f"📄 Loaded {len(candidates)} CSVs for combined report (latest: {csv_path})")

df = pd.read_csv(csv_path)
if "verified_fraction" not in df:
    df["verified_fraction"] = df["verified"] / df["total"]

# === Plots ===
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plot1 = f"plots/trm_verified_fraction_{timestamp}.png"
plt.figure(figsize=(6,4))
plt.plot(df["epsilon"], df["verified_fraction"], "o-", label=df.get("bound", "Bound"))
plt.xlabel("ε (L∞)")
plt.ylabel("Verified Fraction")
plt.title("TRM Certified Robustness")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(plot1, dpi=200)
plt.close()

# === PDF ===
summary = df[["epsilon","verified","verified_fraction","verification_time","memory_usage_mb"]]
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("<b>TRM Robustness Report</b>", styles["Title"]))
story.append(Spacer(1,12))
story.append(Image(plot1, width=400, height=280))
story.append(Spacer(1,12))

data = [summary.columns.tolist()] + summary.values.tolist()
t = Table(data)
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),colors.grey),
    ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
    ('ALIGN',(0,0),(-1,-1),'CENTER'),
    ('GRID',(0,0),(-1,-1),0.5,colors.black)
]))
story.append(t)

out_pdf = f"reports/trm_robustness_report_{timestamp}.pdf"
doc = SimpleDocTemplate(out_pdf, pagesize=A4)
doc.build(story)
print(f"✅ Report saved: {out_pdf}")
