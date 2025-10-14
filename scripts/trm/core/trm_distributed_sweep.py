#!/usr/bin/env python3
"""
Distributed TRM Verification Sweep
Splits sample space across multiple nodes/GPUs using SLURM environment variables
"""

import os
import sys
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
parser.add_argument('--samples', type=int, default=512, help='Total samples to verify')
parser.add_argument('--eps', type=str, required=True, help='Comma-separated epsilon values')
parser.add_argument('--bound', type=str, default='CROWN', help='Bound method')
args = parser.parse_args()

# Get SLURM environment info (with fallbacks for testing)
node_id = int(os.environ.get('SLURM_NODEID', '0'))
num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', '1'))

# Calculate sample range using mentor's formula
input_size = args.samples
start_idx = int(node_id * (input_size / num_nodes))
end_idx = int((node_id + 1) * (input_size / num_nodes))

print(f"=== Distributed TRM Sweep ===")
print(f"Node: {node_id}/{num_nodes}")

# Detect GPUs per node (fallback = 1)
try:
    import torch
    num_gpus_per_node = torch.cuda.device_count()
except Exception:
    num_gpus_per_node = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").count(",") + 1)

# Total parallel workers = nodes × GPUs per node
total_workers = num_nodes * num_gpus_per_node

print(f"GPUs per node: {num_gpus_per_node}")
print(f"Total workers: {total_workers}")
print(f"This worker: samples {start_idx}-{end_idx} ({end_idx - start_idx} samples)")
print(f"Checkpoint: {args.checkpoint}")
print(f"Epsilons: {args.eps}")
print(f"Bound method: {args.bound}")
print()

# Run the actual sweep with sample range
cmd = [
    'python', 'scripts/trm/core/trm_tiny_sweep.py',
    '--checkpoint', args.checkpoint,
    '--samples', str(args.samples),
    '--eps', args.eps,
    '--bound', args.bound,
    '--start-idx', str(start_idx),
    '--end-idx', str(end_idx)
]

print(f"Executing: {' '.join(cmd)}\n")
result = subprocess.run(cmd)
sys.exit(result.returncode)