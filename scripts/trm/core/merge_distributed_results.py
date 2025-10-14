import pandas as pd
import glob

# Find all distributed CSV files
csvs = glob.glob("logs/trm_sweep_CROWN_*.csv")
dfs = [pd.read_csv(f) for f in csvs]
merged = pd.concat(dfs, ignore_index=True)

# Aggregate by epsilon
result = merged.groupby(['model', 'epsilon', 'bound']).agg({
    'verified': 'sum',
    'falsified': 'sum',
    'total': 'sum',
    'avg_time_s': 'mean',
    'avg_mem_MB': 'mean'
}).reset_index()

result.to_csv('logs/trm_sweep_merged.csv', index=False)
print("✅ Merged results saved to logs/trm_sweep_merged.csv")