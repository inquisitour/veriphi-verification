#!/usr/bin/env python3
# scripts/trm/reports/trm_convergence_analysis.py
"""Compare all scaling experiments (64, 256, 512 samples)"""

import pandas as pd
import matplotlib.pyplot as plt

# Load all CSVs
csvs = {
    '64': 'logs/trm_robustness_sweep_v2.csv',
    '256': 'logs/trm_robustness_sweep_v3.csv', 
    '512': 'logs/trm_robustness_sweep_v4.csv'
}

fig, ax = plt.subplots(figsize=(10, 6))

for n, path in csvs.items():
    df = pd.read_csv(path)
    adv = df[df['model'] == 'Adversarial TRM']
    ax.plot(adv['epsilon'], adv['verified']/adv['total'], 
            marker='o', linewidth=2, label=f'n={n}')

ax.set_xlabel('ε (L∞)', fontsize=13)
ax.set_ylabel('Verified Fraction', fontsize=13)
ax.set_title('Convergence Analysis: Sample Size Impact', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('reports/convergence_analysis.png', dpi=200)
print('✅ Convergence plot saved')