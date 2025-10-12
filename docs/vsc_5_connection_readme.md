# VSC5 Cluster Connection Guide (CLI)

This short README explains how to connect to the **VSC5 cluster** via command line (CLI) and prepare your Python + GPU working environment for running the TRM verification stack.

---

## 1️⃣ Prerequisites

Before connecting:
- You must have a **VSC5 user account** and VPN access configured (check that your VPN client is connected).
- You must know your **VSC5 username** and the cluster **hostname** (example: `vsc5.tuwien.ac.at` or a specific login node like `login.vsc5.tuwien.ac.at`).

---

## 2️⃣ Connect to the Cluster via SSH

Open a terminal on your local machine (Linux/Mac or WSL on Windows) and run:

```bash
ssh <your_username>@vsc5.tuwien.ac.at
```

Example:
```bash
ssh fs72936@vsc5.tuwien.ac.at
```

If it’s your first connection, you may need to confirm the host fingerprint.

---

## 3️⃣ Check Available GPU Nodes

Once logged in:

```bash
sinfo -p gpu
```

This lists all GPU partitions and their status. Typical partitions:
- `gpu_a100` — NVIDIA A100 GPUs (recommended for heavy compute)
- `gpu_a40` — NVIDIA A40 GPUs (more VRAM, lower FP64 perf)

To check details of a node:
```bash
scontrol show node <node_name>
```

---

## 4️⃣ Start an Interactive GPU Session

You can request a GPU node interactively (replace partition and time as needed):

```bash
srun --partition=gpu_a100 --gres=gpu:1 --time=02:00:00 --mem=32G --pty bash
```

Explanation:
- `--partition=gpu_a100` → request an A100 GPU node
- `--gres=gpu:1` → 1 GPU
- `--time=02:00:00` → 2 hours
- `--mem=32G` → 32 GB RAM

After this, you will be *inside a GPU compute node shell*.

Check the GPU:
```bash
nvidia-smi
```

---

## 5️⃣ Activate Your Project Environment

From your home or project directory:

```bash
cd ~/veriphi-verification
source venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export VERIPHI_DEVICE=cuda
```

Verify:
```bash
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}:', torch.cuda.get_device_name(i))
PY
```

---

## 6️⃣ Running Jobs Non-Interactively (Optional)

If you want to run a long sweep unattended, submit a Slurm batch job instead of interactive mode.

Example `trm_job.slurm` file:
```bash
#!/bin/bash
#SBATCH --job-name=trm_sweep
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/trm_sweep_%j.out

module load cuda/12.1
source ~/veriphi-verification/venv/bin/activate
cd ~/veriphi-verification
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export VERIPHI_DEVICE=cuda

python scripts/trm_sweep.py --samples 256 --batch 32 --bound alpha-CROWN --opt-steps 100 --log-scaling
```

Submit the job:
```bash
sbatch trm_job.slurm
```

Check status:
```bash
squeue -u <your_username>
```

Cancel a job:
```bash
scancel <job_id>
```

---

## 7️⃣ Disconnecting Safely

When you’re done:
```bash
exit
```
If you were in an interactive `srun` session, that will release the GPU resource automatically.

---

## 8️⃣ Alternative: JupyterHub Access

VSC5 also provides JupyterHub browser access at `https://jupyterhub.vsc.ac.at`:

1. Login with VSC credentials
2. Select GPU resource profile (A100/A40)
3. Launch terminal or notebook
4. Continue from Step 5 above (activate environment)

**Advantage:** No SSH needed, works behind restrictive firewalls.

---

