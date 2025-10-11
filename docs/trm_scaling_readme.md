# TRM Scaling & Benchmark README

This README contains **exact steps** for running the TRM verification sweeps (single-GPU, multi-GPU on one node, and multi-node distributed). It also includes logging, troubleshooting, and a results merge workflow so you can run experiments reproducibly and we can merge the results into the deck.

---

## Prerequisites

1. You (the runner) should have the repo checked out and a working virtualenv.

```bash
git clone https://github.com/inquisitour/veriphi-verification.git
cd veriphi-verification
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

2. Verify core smoke tests pass locally (quick environment sanity):

```bash
python verify_installation.py
python scripts/core_smoke.py
```

3. Make sure checkpoint exists:

```
ls checkpoints/trm_mnist_adv.pt
```

If not present, run `python scripts/trm_tiny_advtrain.py` to produce it.

---

## Repo files you will use

- `scripts/trm_tiny_sweep.py` — main sweep driver (we will add/patch a few flags)
- `checkpoints/trm_mnist_adv.pt` — model to verify
- `data/scaling/` — directory where each run will write CSV logs
- `scripts/trm_tiny_verify.py` — for test runs

---

## Small required edits (one-time)

Open `scripts/trm/trm_tiny_sweep.py` and add the following helper logging and flags near the argument parser: (copy/paste)

```python
# add imports
import time, csv, os
from datetime import datetime

# add optional args to parser
parser.add_argument('--samples', type=int, default=256, help='number of samples to verify')
parser.add_argument('--batch', type=int, default=32, help='batch size for verification')
parser.add_argument('--bound', type=str, default='alpha-CROWN', help='bound method')
parser.add_argument('--opt-steps', type=int, default=50, help='optimization steps for alpha/beta crown')
parser.add_argument('--log-scaling', action='store_true', help='write results to data/scaling CSV')
```

Add a small CSV writer function near the top of the script:

```python
def write_scaling_log(outfile, row):
    header = [
        'timestamp','config','devices','batch','samples','avg_time_per_sample_s','total_time_s','gpu_mem_mb','notes'
    ]
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    write_header = not os.path.exists(outfile)
    with open(outfile, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
```

Use this writer to write a result row at the end of the run when `--log-scaling` is set.

---

## Commands & step-by-step tasks

### Tasks

- **Kartik** — Single GPU baseline (+ optional multi-node later)
- **Vassi** — Two GPUs on the same node using DataParallel

If either of you can also run the multi-node torchrun command, that would be great — just coordinate who requests the 2-node reservation from Claudia.

---

### 1) Single GPU baseline (Kartik)

**Goal:** confirm baseline numbers (time/sample ~0.22s and GPU mem ~440 MB)

Commands to run (copy-paste):

```bash
# Ensure one GPU is visible
export CUDA_VISIBLE_DEVICES=0
export VERIPHI_DEVICE=cuda:0
source venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# Run sweep
python scripts/trm_tiny_sweep.py --samples 256 --batch 32 --bound alpha-CROWN --opt-steps 100 --log-scaling
```

Notes for Kartik:
- Run 2× (early and repeat) to ensure stability. If runs differ >10%, rerun.
- Capture GPU memory: while the run is executing, execute `nvidia-smi --query-gpu=memory.used --format=csv -l 1` in another terminal and keep 1–2 samples of peak memory printed.
- The script will log a CSV into `data/scaling/scaling_<timestamp>.csv` when `--log-scaling` is used.

Expected outputs (example):
```
Avg time/sample: 0.22 s
Peak GPU mem: ~440 MB
```

If you see CUDA OOM, reduce `--batch` to 16 and rerun.

---

### 2) Two GPUs, same node (Vassi)

**Goal:** check speedup and memory scaling using `torch.nn.DataParallel`.

Edit `scripts/trm_tiny_sweep.py` near model instantiation (one-line change):

```python
# after `model = create_model(...)`
model = torch.nn.DataParallel(model)
```

Commands to run (copy-paste):

```bash
export CUDA_VISIBLE_DEVICES=0,1
export VERIPHI_DEVICE=cuda
source venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"

python scripts/trm_tiny_sweep.py --samples 256 --batch 64 --bound alpha-CROWN --opt-steps 100 --log-scaling
```

Notes for Vassi:
- Use batch 64 to better utilize both GPUs; drop to batch 32 if you see OOMs.
- Capture memory per-GPU via `nvidia-smi`. You should see roughly double memory footprint across GPUs.
- Run 2× and record both runs.

Expected (rough) results:
```
Avg time/sample: ~0.12 s (near-linear speedup compared to single GPU baseline)
GPU mem per device: ~440 MB each (total ~880 MB)
```

---

### 3) Multi-node (optional / coordinate with me)

**Goal:** run distributed torchrun across 2 nodes (1 GPU each) and measure end-to-end throughput.

You will need:
- Reservation for 2 nodes (A100 or A40 as requested).
- One node as rendezvous host: `<master_ip>:29500` (use the node's internal IP).

Launch (example):

```bash
# On the master node (or via job submission):
torchrun --nnodes=2 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=<master_ip>:29500 \
  scripts/trm_tiny_sweep.py --samples 512 --batch 64 --bound alpha-CROWN --opt-steps 100 --log-scaling
```

Notes:
- Each process will verify its shard of samples; the script should be updated to aggregate statistics (avg time/sample across processes). We can merge logs from each node afterwards.
- Use `nvidia-smi` on both nodes to capture memory.

---

## Logging format (CSV)

Each run should append a single row to `data/scaling/scaling_<timestamp>.csv` with these columns:
```
timestamp,config,devices,batch,samples,avg_time_per_sample_s,total_time_s,gpu_mem_mb,notes
```

Example row:
```
2025-10-11T21:10:00Z,alpha-CROWN,1xA100,32,256,0.220,56.32,440,baseline run
```

---

## How to merge results and add to the deck

1. After runs are complete, push CSVs to the repo branch (or central team Drive): `data/scaling/`.
2. I will run `scripts/plot_scaling.py` (I can supply) to produce the summary plots and numbers for the slide.
3. Update slides: add the table + 1 plot showing `time/sample` vs `#GPUs` and memory usage.

---

## Troubleshooting common errors

- **CUDA OOM**: reduce `--batch`, remove DataParallel (run single GPU to debug), or lower `--opt-steps`.
- **Streamlit unreachable**: If running streamlit on remote via VS Code SSH tunnel, ensure ports are forwarded (see note below).
- **Different auto-LiRPA versions**: if you see `compute_bounds()` signature errors, ensure auto-LiRPA is pinned to the project version: `pip install -e ./auto_LiRPA` (v0.6.0 recommended).

---

## Port-forwarding note (streamlit)

If you run streamlit on a remote machine and want to access via your laptop browser while connected over VPN / SSH:

- Create an SSH tunnel from your laptop:

```bash
ssh -L 8501:localhost:8501 your_user@remote_host
```

Then open `http://localhost:8501` on your laptop.

If using VS Code Remote SSH, use the built-in port forwarding in the Remote Explorer.

---

