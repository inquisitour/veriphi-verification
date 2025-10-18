'''#!/usr/bin/env python3
# scripts/trm/reports/trm_convergence_analysis.py
"""Compare all scaling experiments (64, 256, 512 samples)"""

import pandas as pd
import matplotlib.pyplot as plt

# Load all CSVs
'''csvs = {
    '64': 'logs/trm_robustness_sweep_v2.csv',
    '256': 'logs/trm_robustness_sweep_v3.csv', 
    '512': 'logs/trm_robustness_sweep_v4.csv'
}'''

csvs = {
    '64': 'logs/trm_sweep_.csv',
    '256': 'logs/trm_sweep_.csv', 
    '512': 'logs/trm_sweep_.csv'
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
print('✅ Convergence plot saved')'''


#!/usr/bin/env python3
"""
Convergence Analysis: How verification accuracy stabilizes with sample size
"""

import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-base', type=str, required=True, 
                   help='Model base name (e.g., trm_mnist_ibp_eps015)')
parser.add_argument('--bound', type=str, default='CROWN',
                   help='Bound method to analyze')
parser.add_argument('--output-dir', type=str, default='reports',
                   help='Output directory for plots')
args = parser.parse_args()

def get_convergence_csvs(model_base, bound='CROWN'):
    """Find all convergence CSVs for a model and bound method"""
    pattern = f"logs/trm_sweep_{model_base}_{bound}_0_*.csv"
    files = glob.glob(pattern)
    
    csvs = {}
    for f in files:
        match = re.search(r'_0_(\d+)\.csv$', f)
        if match:
            sample_size = match.group(1)
            csvs[sample_size] = f
    
    return csvs

# Get CSVs
csvs = get_convergence_csvs(args.model_base, args.bound)

if not csvs:
    print(f"❌ No CSVs found for model: {args.model_base}, bound: {args.bound}")
    print(f"   Looking for pattern: logs/trm_sweep_{args.model_base}_{args.bound}_0_*.csv")
    exit(1)

print(f"Found {len(csvs)} sample sizes: {sorted(csvs.keys())}")

# Load and analyze data
results = {}
for n, csv_path in csvs.items():
    df = pd.read_csv(csv_path)
    
    # Group by epsilon and calculate verified fraction
    grouped = df.groupby('epsilon').agg({
        'verified': 'sum',
        'total': 'sum'
    }).reset_index()
    
    grouped['verified_fraction'] = grouped['verified'] / grouped['total']
    results[n] = grouped

# Plot convergence
plt.figure(figsize=(10, 6))
for n in sorted(results.keys(), key=int):
    data = results[n]
    plt.plot(data['epsilon'], data['verified_fraction'], 
            marker='o', label=f'n={n}')

plt.xlabel('Epsilon (ε)')
plt.ylabel('Verified Fraction')
plt.title(f'Convergence Analysis: {args.model_base} ({args.bound})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = f"{args.output_dir}/convergence_{args.model_base}_{args.bound}.png"
plt.savefig(output_path, dpi=150)
print(f"✅ Saved: {output_path}")