#!/bin/bash
#SBATCH --job-name=trm_distributed
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH -N 2
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00    # 1 day
#SBATCH --job-name=trm_multinode
#SBATCH -o logs/%x_%j.out
#SBATCH --error=logs/distributed_%j_%N.err

cd $HOME/veriphi-verification
source venv/bin/activate
export VERIPHI_DEVICE=cuda
export PYTHONPATH="$PWD/src:$PYTHONPATH"

echo "Job started at $(date)"
echo "Node: $SLURM_NODEID / $SLURM_NNODES"
echo "GPUs: $SLURM_GPUS_ON_NODE"

# Run distributed sweep
srun python scripts/trm/core/trm_distributed_sweep.py \
  --checkpoint checkpoints/trm_mnist_adv_eps015.pt \
  --samples 512 \
  --eps "0.02,0.1,0.3" \
  --bound CROWN

echo "Job finished at $(date)"