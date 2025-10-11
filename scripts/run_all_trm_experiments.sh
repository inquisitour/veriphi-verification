#!/bin/bash
# ============================================================
# Run all TRM verification experiments (CROWN, α, β, heavy)
# ============================================================

set -e
timestamp=$(date +"%Y%m%d_%H%M%S")
device=${VERIPHI_DEVICE:-cuda}
echo "🚀 Starting TRM experiments @ $timestamp (Device=$device)"

mkdir -p runs logs plots reports

SWEEP="scripts/trm/core/trm_tiny_sweep.py"
CHECKPOINT="checkpoints/trm_mnist_adv.pt"

# Helper to add per-experiment timestamps
ts() { date +"%Y%m%d_%H%M%S"; }

# 1️⃣ --- Baseline: CROWN ---
echo "🔹 [1/5] Running CROWN baseline..."
python $SWEEP \
  --checkpoint $CHECKPOINT \
  --eps 0.03,0.05,0.1 \
  --bound CROWN \
  --samples 64 --batch 16 \
  --verify-timeout 60 \
  --out runs/trm_sweep_crown_$(ts).csv | tee logs/exp1_crown_$(ts).log

# 2️⃣ --- Alpha-CROWN ---
echo "🔹 [2/5] Running α-CROWN baseline..."
python $SWEEP \
  --checkpoint $CHECKPOINT \
  --eps 0.03,0.05,0.1 \
  --bound alpha-CROWN \
  --samples 64 --batch 16 \
  --opt-steps 50 --lr 0.01 \
  --verify-timeout 90 \
  --out runs/trm_sweep_alpha_$(ts).csv | tee logs/exp2_alpha_$(ts).log

# 3️⃣ --- Beta-CROWN ---
echo "🔹 [3/5] Running β-CROWN baseline..."
python $SWEEP \
  --checkpoint $CHECKPOINT \
  --eps 0.03,0.05,0.1 \
  --bound beta-CROWN \
  --samples 64 --batch 16 \
  --opt-steps 50 --lr 0.01 \
  --verify-timeout 90 \
  --out runs/trm_sweep_beta_$(ts).csv | tee logs/exp3_beta_$(ts).log

# 4️⃣ --- Heavy Experiment A (α-CROWN long opt) ---
echo "🔹 [4/5] Running Experiment A (heavy α-CROWN)..."
python $SWEEP \
  --checkpoint $CHECKPOINT \
  --eps 0.03,0.05,0.10 \
  --bound alpha-CROWN \
  --samples 256 --batch 32 \
  --opt-steps 150 --lr 0.01 \
  --verify-timeout 120 \
  --out runs/trm_sweep_alpha_heavyA_$(ts).csv | tee logs/exp4_alpha_heavyA_$(ts).log

# 5️⃣ --- Heavy Experiment B (larger batch + longer opt) ---
echo "🔹 [5/5] Running Experiment B (extra heavy α-CROWN)..."
python $SWEEP \
  --checkpoint $CHECKPOINT \
  --eps 0.03,0.05 \
  --bound alpha-CROWN \
  --samples 512 --batch 64 \
  --opt-steps 200 --lr 0.01 \
  --verify-timeout 180 \
  --out runs/trm_sweep_alpha_heavyB_$(ts).csv | tee logs/exp5_alpha_heavyB_$(ts).log

# --- Consolidate all results ---
echo "📊 Generating TRM robustness report..."
python scripts/trm/reports/trm_generate_report.py || echo "⚠️ Report generation skipped (no CSVs?)"

echo "✅ All TRM experiments complete!"
echo "Results saved under runs/, logs/, plots/, and reports/"
