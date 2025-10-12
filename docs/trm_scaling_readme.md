# TRM Scaling & Benchmark README

**Exact steps for running TRM verification sweeps across single-GPU, multi-GPU (same node), and multi-node distributed setups on VSC5 A100 clusters.**

---

## Prerequisites

1. **Clone and setup environment:**

```bash
git clone https://github.com/inquisitour/veriphi-verification.git
cd veriphi-verification
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

2. **Verify installation:**

```bash
python verify_installation.py
python scripts/core_smoke.py
```

3. **Ensure checkpoints exist:**

```bash
ls checkpoints/trm_mnist.pt         # Standard TRM
ls checkpoints/trm_mnist_adv.pt     # Adversarial TRM (recommended)
```

If missing, generate them:
```bash
python scripts/trm/core/trm_tiny_train.py       # Standard
python scripts/trm/core/trm_tiny_advtrain.py    # Adversarial
```

---

## Repository Structure

```
scripts/trm/
├── core/
│   ├── trm_tiny_train.py        # Train standard TRM
│   ├── trm_tiny_advtrain.py     # Train adversarial TRM
│   ├── trm_tiny_verify.py       # Single-sample verification
│   └── trm_tiny_sweep.py        # Batch verification sweep ⭐
├── reports/
│   ├── trm_visualize_results.py      # Generate plots
│   ├── trm_full_visual_report.py     # Generate PDF report
│   └── trm_convergence_analysis.py   # Convergence analysis
└── presentation/
    └── trm_presentation_slide.py     # Generate PowerPoint

checkpoints/          # Model checkpoints
logs/                 # CSV sweep results
plots/                # Generated visualizations
reports/              # PDF reports and presentations
data/scaling/         # Scaling experiment logs (create if needed)
```

---

## Task Assignments

### **Kartik** — Single GPU Baseline
- Run 512-sample sweep on 1×A100
- Establish baseline timing/memory metrics
- Optional: coordinate multi-node experiment

### **Vassi** — Multi-GPU Same Node
- Run sweep on 2×A100 using parallel execution
- Compare throughput vs single GPU
- Document memory scaling across GPUs

### **Coordination**
- For multi-node: coordinate with me for 2-node allocation
- Share results in `data/scaling/` with descriptive filenames
- Run experiments 2× each for stability verification

---

## Sweep Script Interface

`scripts/trm/core/trm_tiny_sweep.py` accepts:

```bash
--samples N          # Number of test samples (default: 20)
--checkpoint PATH    # Model checkpoint (default: both standard + adversarial)
--eps VALUES         # Comma-separated epsilon values (default: 0.01,0.02,0.03,0.04,0.06,0.08,0.1)
--timeout SECONDS    # Per-sample timeout (default: 60)
```

**Current baseline (512 samples, 1×A100):**
- Time/sample: ~0.17s (standard), ~0.20s (adversarial)
- GPU memory: ~18-30 MB
- Verified: 80% @ ε=0.01 (adversarial), 1% @ ε=0.01 (standard)

---

## Task 1: Single GPU Baseline (Kartik)

**Goal:** Confirm baseline performance metrics (time/sample ~0.20s, GPU mem ~28 MB).

**Setup:**
```bash
export CUDA_VISIBLE_DEVICES=0
export VERIPHI_DEVICE=cuda
source venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

**Commands:**

```bash
# Quick validation (64 samples, ~1 min)
python scripts/trm/core/trm_tiny_sweep.py --samples 64

# Standard benchmark (256 samples, ~4 min)
python scripts/trm/core/trm_tiny_sweep.py --samples 256

# Full baseline (512 samples, ~8 min)
python scripts/trm/core/trm_tiny_sweep.py --samples 512
```

**Capture GPU metrics:**
```bash
# In separate terminal during run:
watch -n 1 nvidia-smi
# Record peak memory usage from display
```

**Deliverables:**
- Run each config 2× to verify stability (difference <10%)
- Record to `data/scaling/kartik_1gpu_baseline.csv`:
  ```
  timestamp,config,devices,batch,samples,avg_time_per_sample_s,total_time_s,gpu_mem_mb,notes
  2025-10-13T14:30:00,alpha-CROWN,1xA100,10,512,0.197,101.0,28.5,run1
  ```
- Save CSV outputs from `logs/` directory

**Expected results:**
- Avg time/sample: 0.17-0.20s
- GPU memory: 18-30 MB
- If CUDA OOM: reduce samples, report issue

---

## Task 2: Multi-GPU Same Node (Vassi)

**Goal:** Measure speedup with 2×A100 parallel execution.

**Approach:** Run two instances in parallel, each on separate GPU.

**Terminal 1:**
```bash
export CUDA_VISIBLE_DEVICES=0
export VERIPHI_DEVICE=cuda
source venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"

python scripts/trm/core/trm_tiny_sweep.py --samples 256 > logs/vassi_gpu0.log 2>&1
```

**Terminal 2 (simultaneously):**
```bash
export CUDA_VISIBLE_DEVICES=1
export VERIPHI_DEVICE=cuda
source venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"

python scripts/trm/core/trm_tiny_sweep.py --samples 256 > logs/vassi_gpu1.log 2>&1
```

**Monitor both GPUs:**
```bash
watch -n 1 nvidia-smi
```

**Deliverables:**
- Run 2× for consistency
- Record to `data/scaling/vassi_2gpu_parallel.csv`:
  ```
  timestamp,config,devices,batch,samples,avg_time_per_sample_s,total_time_s,gpu_mem_mb,notes
  2025-10-13T15:00:00,alpha-CROWN,2xA100,10,512,0.10,51.2,28.5_per_gpu,parallel_split
  ```
- Calculate total throughput: (samples_gpu0 + samples_gpu1) / max(time_gpu0, time_gpu1)
- Note memory per GPU from nvidia-smi

**Expected results:**
- Near 2× throughput vs single GPU
- Memory: ~28 MB per GPU (similar to single GPU)
- Total wallclock time: ~50% of single GPU run

---

## Task 3: Multi-Node Distributed

**Goal:** Test scaling across 2 compute nodes (1 GPU each).

**Requirements:**
- Request 2-node allocation from cluster admin
- Both nodes must have network connectivity
- Coordinate launch timing

**On master node:**

```bash
# Get master node IP
MASTER_IP=$(hostname -I | awk '{print $1}')
echo "Master IP: $MASTER_IP"

export MASTER_ADDR=$MASTER_IP
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=0

torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  scripts/trm/core/trm_tiny_sweep.py --samples 512
```

**On worker node (share master IP privately):**

```bash
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=1

torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  scripts/trm/core/trm_tiny_sweep.py --samples 512
```

**Note:** Current script needs distributed data loading modifications. Start both within 5 minutes for rendezvous success.

**Deliverables:**
- Record combined throughput
- Note any synchronization overhead
- Document in `data/scaling/multinode_2x1gpu.csv`

---

## Slurm Batch Jobs Example (Alternative to Interactive)

**Create** `jobs/trm_baseline.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=trm_1gpu
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/kartik_%j.out

module load cuda/12.1
source ~/veriphi-verification/venv/bin/activate
cd ~/veriphi-verification
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export VERIPHI_DEVICE=cuda

python scripts/trm/core/trm_tiny_sweep.py --samples 512
```

**Submit:**
```bash
sbatch jobs/trm_baseline.slurm
```

---

## Results Merge & Analysis

**After experiments complete:**

1. **Push CSVs to repo:**
   ```bash
   git add data/scaling/*.csv logs/*.csv
   git commit -m "Add scaling benchmarks: 1GPU, 2GPU results"
   git push
   ```

2. **Generate visualizations:**
   ```bash
   python scripts/trm/reports/trm_visualize_results.py
   python scripts/trm/reports/trm_convergence_analysis.py
   ```

3. **Create comparison plot:**
   - Parse `data/scaling/*.csv`
   - Plot time/sample vs GPU count
   - Show memory scaling
   - Add to presentation deck

---

## Logging Format

**Sweep outputs:** `logs/trm_robustness_sweep_v*.csv`
```
model,epsilon,verified,falsified,total,avg_time_s,avg_mem_MB
```

**Scaling logs:** `data/scaling/scaling_*.csv`
```
timestamp,config,devices,batch,samples,avg_time_per_sample_s,total_time_s,gpu_mem_mb,notes
```

**Naming convention:**
- `data/scaling/kartik_1gpu_run1.csv`
- `data/scaling/vassi_2gpu_parallel_run1.csv`
- `data/scaling/multinode_2x1gpu_run1.csv`

---

## Troubleshooting

**CUDA OOM:** Reduce samples.

**Slow verification:** Check `nvidia-smi` shows GPU utilization >80%.

**Import errors:** Ensure `export PYTHONPATH="$PWD/src:$PYTHONPATH"` is set.

**Multi-node timeout:** Launch both nodes within 5 minutes. Check firewall allows port 29500.

---

**Last updated:** 2025-10-13