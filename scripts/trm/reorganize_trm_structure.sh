#!/usr/bin/env bash
# ============================================================
#  Veriphi TRM Project Cleanup & Safe Reorganization Script
#  — Keeps venv active, never replaces shell
# ============================================================

set -euo pipefail
echo "🧩 Starting TRM structure reorganization (safe mode)..."

# ------------------------------------------------------------
# 1️⃣ Resolve absolute paths
# ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"
TRM_DIR="${REPO_ROOT}/scripts/trm"

cd "${REPO_ROOT}"
echo "📍 Working directory: ${REPO_ROOT}"

if [[ ! -d "${TRM_DIR}" ]]; then
  echo "❌ Error: TRM directory not found at ${TRM_DIR}"
  exit 1
fi

# ------------------------------------------------------------
# 2️⃣ Create structured layout (idempotent)
# ------------------------------------------------------------
echo "📁 Creating subdirectories..."
mkdir -p "${TRM_DIR}/core" "${TRM_DIR}/reports" "${TRM_DIR}/presentation" "${TRM_DIR}/misc"
mkdir -p logs runs plots reports checkpoints

# ------------------------------------------------------------
# 3️⃣ Move files safely (never override existing)
# ------------------------------------------------------------
move_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "${src}" ]]; then
    echo "→ Moving ${src} → ${dst}/"
    mv -n "${src}" "${dst}/" 2>/dev/null || true
  fi
}

echo "🚀 Moving core experiment scripts..."
move_if_exists "${TRM_DIR}/trm_tiny_train.py"        "${TRM_DIR}/core"
move_if_exists "${TRM_DIR}/trm_tiny_advtrain.py"     "${TRM_DIR}/core"
move_if_exists "${TRM_DIR}/trm_tiny_verify.py"       "${TRM_DIR}/core"
move_if_exists "${TRM_DIR}/trm_tiny_sweep.py"        "${TRM_DIR}/core"

echo "📊 Moving reporting and visualization scripts..."
move_if_exists "${TRM_DIR}/trm_visualize_results.py" "${TRM_DIR}/reports"
move_if_exists "${TRM_DIR}/trm_generate_report.py"   "${TRM_DIR}/reports"
move_if_exists "${TRM_DIR}/trm_full_visual_report.py" "${TRM_DIR}/reports"
move_if_exists "${TRM_DIR}/trm_compare_bounds_report.py" "${TRM_DIR}/reports"

echo "🖼️ Moving presentation and misc scripts..."
move_if_exists "${TRM_DIR}/trm_presentation_slide.py" "${TRM_DIR}/presentation"
move_if_exists "${TRM_DIR}/trm_smoke.py"              "${TRM_DIR}/misc"

# ------------------------------------------------------------
# 4️⃣ Generate README.md dynamically
# ------------------------------------------------------------
echo "🧾 Generating TRM README.md..."
cat > "${TRM_DIR}/README.md" <<EOF
# 🧠 Veriphi TRM Suite — Modular Structure Overview

This directory contains all scripts for **TRM (Transformer-style Robust Model)** verification, training, and visualization.

Organized for clarity, GPU scaling, and reproducibility.

---

## 📁 Folder Layout

\`\`\`bash
scripts/trm/
├── core/              # Training & verification logic
├── reports/           # Plots and PDF generation
├── presentation/      # Hackathon slide deck generator
└── misc/              # Sanity tests & utilities
\`\`\`

---

### 🚀 Usage Summary

#### Train baseline TRM
\`\`\`bash
python scripts/trm/core/trm_tiny_train.py
\`\`\`

#### Train adversarial TRM
\`\`\`bash
python scripts/trm/core/trm_tiny_advtrain.py
# Output: checkpoints/trm_mnist_adv.pt
\`\`\`

#### Verify robustness (α-CROWN)
\`\`\`bash
python scripts/trm/core/trm_tiny_verify.py --bound alpha-CROWN --eps 0.03
\`\`\`

#### Run sweep
\`\`\`bash
python scripts/trm/core/trm_tiny_sweep.py \
  --checkpoint checkpoints/trm_mnist_adv.pt \
  --eps 0.03,0.05,0.10 --norm inf \
  --samples 256 --batch 32 \
  --bound alpha-CROWN --opt-steps 150 --lr 0.01
\`\`\`

---

### 📈 Visualization & Reporting
\`\`\`bash
python scripts/trm/reports/trm_generate_report.py
python scripts/trm/reports/trm_full_visual_report.py
python scripts/trm/reports/trm_compare_bounds_report.py
\`\`\`

Plots → \`plots/\`  
PDFs  → \`reports/\`

---

### 🧪 Smoke Test
\`\`\`bash
python scripts/trm/misc/trm_smoke.py
\`\`\`

Expected output:
✓ Attack-guided verification engine initialized on cuda
Verification result: verified ...


---
 
**Date:** $(date +%Y-%m-%d)
EOF

echo "✅ README.md generated at: ${TRM_DIR}/README.md"

# ------------------------------------------------------------
# 5️⃣ Verify imports (non-fatal)
# ------------------------------------------------------------
echo "🧠 Verifying imports (non-fatal)..."
PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/scripts" \
python - <<'PY'
import importlib
modules = [
"trm.core.trm_tiny_train",
"trm.core.trm_tiny_advtrain",
"trm.core.trm_tiny_verify",
"trm.core.trm_tiny_sweep",
"trm.reports.trm_visualize_results",
"trm.reports.trm_generate_report",
"trm.reports.trm_full_visual_report",
"trm.reports.trm_compare_bounds_report",
"trm.presentation.trm_presentation_slide",
"trm.misc.trm_smoke",
]
for m in modules:
    try:
        importlib.import_module(m)
        print(f"✅ Import OK: {m}")
    except Exception as e:
        print(f"⚠️  Import failed: {m} — {e}")
PY

echo "🎯 TRM reorganization complete! Virtual env preserved."
