#!/bin/bash
#SBATCH --job-name=trm_distributed
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH -N 2
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00    # 1 day
#SBATCH --job-name=trm_ibp_convergence_singlenode
#SBATCH -o logs/ibp_convergence_%j.out
#SBATCH --error=logs/ibp_convergence_%j.err


cd $HOME/veriphi-verification
source venv/bin/activate
export VERIPHI_DEVICE=cuda
export PYTHONPATH="$PWD/src:$PYTHONPATH"

echo "=== Convergence Analysis Started (All Bounds) ==="
echo "Job started at $(date)"

# Loop through all bounds
for BOUND in CROWN alpha-CROWN beta-CROWN; do
  echo ""
  echo "=== Testing bound: $BOUND ==="
  
  # IBP @ ε=0.15
  for n in 64 128 256; do
    echo "  IBP@0.15 with n=$n samples ($BOUND)"
    CUDA_VISIBLE_DEVICES=0 python scripts/trm/core/trm_tiny_sweep.py \
      --checkpoint checkpoints/trm_mnist_ibp_eps015_weights.pt \
      --samples $n --eps "0.05,0.1,0.15" --bound $BOUND \
      | tee logs/trm_sweep_trm_mnist_ibp_eps015_${BOUND}_0_${n}_$(date +%Y%m%d_%H%M%S).log
  done
  
  # IBP @ ε=0.01
  for n in 64 128 256; do
    echo "  IBP@0.01 with n=$n samples ($BOUND)"
    CUDA_VISIBLE_DEVICES=1 python scripts/trm/core/trm_tiny_sweep.py \
      --checkpoint checkpoints/trm_mnist_ibp_eps001_weights.pt \
      --samples $n --eps "0.01,0.05,0.1" --bound $BOUND \
      | tee logs/trm_sweep_trm_mnist_ibp_eps001_${BOUND}_0_${n}_$(date +%Y%m%d_%H%M%S).log
  done
done

echo ""
echo "✅ Convergence analysis complete for all bounds"
echo "Job finished at $(date)"