#!/bin/bash
#SBATCH -A p72936
#SBATCH -p zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:2
#SBATCH -t 00:20:00
#SBATCH -J veriphi_gpu
#SBATCH -o logs/gpu_%j.out
#SBATCH -e logs/gpu_%j.err

source ~/venv/bin/activate
cd ~/veriphi-verification
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export VERIPHI_DEVICE=cuda     
export CUDA_VISIBLE_DEVICES=0  
export OMP_NUM_THREADS=1
nvidia-smi
module load cuda/12.9.0-gcc-12.2.0-xznhx43
python verify_installation.py
python scripts/core_smoke.py
python scripts/trm/misc/trm_smoke.py