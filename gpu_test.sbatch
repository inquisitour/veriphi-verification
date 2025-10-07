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
module load cuda/12.9.0-gcc-12.2.0-xznhx43
pip install -q git+https://github.com/Verified-Intelligence/auto_LiRPA.git
python verify_installation.py
