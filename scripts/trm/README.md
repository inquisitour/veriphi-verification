# 🧠 Veriphi TRM Suite — Modular Structure Overview

This directory contains all scripts for **TRM (Transformer-style Robust Model)** verification, training, and visualization.

Organized for clarity, GPU scaling, and reproducibility.

---

## 📁 Folder Layout

```bash
scripts/trm/
├── core/              # Training & verification logic
├── reports/           # Plots and PDF generation
├── presentation/      # Hackathon slide deck generator
└── misc/              # Sanity tests & utilities
```

---

### 🚀 Usage Summary

#### Train baseline TRM
```bash
python scripts/trm/core/trm_tiny_train.py
```

#### Train adversarial TRM
```bash
python scripts/trm/core/trm_tiny_advtrain.py
# Output: checkpoints/trm_mnist_adv.pt
```

#### Verify robustness (α-CROWN)
```bash
python scripts/trm/core/trm_tiny_verify.py --bound alpha-CROWN --eps 0.03
```

#### Run sweep
```bash
python scripts/trm/core/trm_tiny_sweep.py   --checkpoint checkpoints/trm_mnist_adv.pt   --eps 0.03,0.05,0.10 --norm inf   --samples 256 --batch 32   --bound alpha-CROWN --opt-steps 150 --lr 0.01
```

---

### 📈 Visualization & Reporting
```bash
python scripts/trm/reports/trm_generate_report.py
python scripts/trm/reports/trm_full_visual_report.py
python scripts/trm/reports/trm_compare_bounds_report.py
```

Plots → `plots/`  
PDFs  → `reports/`

---

### 🧪 Smoke Test
```bash
python scripts/trm/misc/trm_smoke.py
```

Expected output:
✓ Attack-guided verification engine initialized on cuda
Verification result: verified ...


---
 
**Date:** 2025-10-11
