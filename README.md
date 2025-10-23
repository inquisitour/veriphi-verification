# 🧠 Veriphi: Neural Network Robustness Verification

A **GPU‑accelerated verification stack** combining **attack‑guided adversarial search** with **formal bound certification**  
(α‑, β‑CROWN via [auto‑LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA)).

Developed at **OpenACC Hackathon 2025** by Team Veriphi (TU Wien).

It answers a simple but critical question:

> **"Is this model provably robust within ε under L∞ or L2 perturbations?"**

…and returns **verified / falsified**, with measured **runtime & memory**.

---

## 🚀 Highlights

✅ **Attack‑Guided Verification:**  
   Fast falsification via FGSM + I‑FGSM (85% time reduction), then formal verification using α‑, β‑CROWN.

✅ **Production Scale:**  
   Successfully verified **105.8M parameter TRM** on real Airbus Beluga logistics (270 problems, 2.6s/sample).

✅ **TRM‑MLP Integration:**  
   Support for **Tiny Recursive Models (TRM)** — verified using the same unified pipeline.

✅ **GPU‑Accelerated Verification:**  
   Works seamlessly on **A100, RTX** or any CUDA‑enabled GPU. **5× speedup** over CPU baseline.

✅ **Cross-Dataset Validation:**  
   Comprehensive verification on **MNIST** and **CIFAR-10** with 3 training methods (Baseline, IBP, PGD).

✅ **Multi-Bound Comparison:**  
   Systematic evaluation of **CROWN, α-CROWN, β-CROWN** across datasets and models.

✅ **Research Finding:**  
   Training method effectiveness depends on dataset complexity—IBP excels on simple data, PGD dominates complex data.

---

## 🔧 Install (reproducible)

```bash
# Clone
git clone https://github.com/inquisitour/veriphi-verification.git
cd veriphi-verification

# Python env
python3 -m venv venv
source venv/bin/activate

# Installation
pip install git+https://github.com/Verified-Intelligence/auto_LiRPA.git
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

Verify your toolchain:
```bash
python verify_installation.py
```

---

## 🚀 Quick smoke test

```bash
# From repo root
source venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python scripts/core_smoke.py
```

---

## ⚡ GPU mode

Veriphi fully supports **CUDA (A100, RTX, etc.)**.

```bash
# Enable GPU device
export VERIPHI_DEVICE=cuda
```

All engines, attacks, and models will automatically run on the GPU.

---

## 📊 Complete Verification Results

### MNIST (28×28 grayscale, 512 samples)

| Training Method | ε=0.04 | ε=0.06 | ε=0.08 | ε=0.1 | **Winner** |
|----------------|--------|--------|--------|-------|------------|
| **IBP (1/255)** | 47% | **77%** | **78%** | **75%** | 🥇 |
| **PGD (2/255)** | 43% | 63% | 65% | 60% | 🥈 |
| **Baseline** | 0% | 0% | 0% | 0% | ❌ |

### CIFAR-10 (32×32 RGB, 512 samples)

| Training Method | ε=0.001 | ε=0.002 | ε=0.004 | ε=0.006 | **Winner** |
|----------------|---------|---------|---------|---------|------------|
| **PGD (8/255)** | **94%** | **90%** | **80%** | **67%** | 🥇 |
| **IBP (2/255)** | 78% | 51% | 10% | 1% | 🥈 |
| **Baseline** | 82% | 55% | 13% | 1% | 🥉 |

**Key Finding:** Training method effectiveness depends on dataset complexity:
- **IBP dominates on simple MNIST** (75-78% @ ε=0.06-0.1)
- **PGD dominates on complex CIFAR-10** (48-95% across all ε)
- **Bound methods (α/β-CROWN)** provide <5% improvement over CROWN

---

## 🚀 Production Scale: Airbus Beluga Logistics

Successfully scaled to **105.8M parameter TRM** on real-world constraint satisfaction:

**Dataset:** 270 Airbus Beluga logistics problems from TUPLES AI Challenge
- 69-821 jigs per problem
- 43-199 flights per problem
- 5 constraint types (capacity, jig matching, type matching, exclusivity, multi-trip)

**Performance:**
- Verification: **2.6s per sample** on A100
- Training loss: 930 → 2.26
- GPU memory: Efficient scaling to 105M parameters
- Successfully profiled with Nsight Systems

**Achievement:** First-ever formal verification of constraint satisfaction model at this scale.

---

## 🧩 TRM Experiments

### Training Scripts

```bash
# MNIST
python scripts/trm/core/trm_tiny_train.py              # Baseline
python scripts/trm/core/trm_tiny_advtrain.py           # PGD adversarial
python scripts/trm/core/trm_ibp_train.py               # IBP certified

# CIFAR-10
python scripts/trm/core/trm_tiny_train_cifar10.py      # Baseline
python scripts/trm/core/trm_tiny_advtrain_cifar10.py   # PGD adversarial
python scripts/trm/core/trm_ibp_train_cifar10.py       # IBP certified
```

### Verification Sweeps

```bash
# MNIST - Full sweep (512 samples, 3 bounds)
python scripts/trm/core/trm_tiny_sweep.py \
  --samples 512 \
  --eps 0.01,0.02,0.03,0.04,0.06,0.08,0.1 \
  --bound CROWN

# CIFAR-10 - Full sweep
python scripts/trm/core/trm_tiny_sweep_cifar10.py \
  --samples 512 \
  --eps 0.001,0.002,0.004,0.006,0.008,0.01 \
  --bound CROWN
```

### Generate Reports

```bash
# MNIST comprehensive report
python scripts/trm/reports/trm_full_visual_report_mnist.py

# CIFAR-10 comprehensive report
python scripts/trm/reports/trm_full_visual_report_cifar10.py

# Bound comparison across datasets
python scripts/trm/reports/trm_compare_bounds_report.py
```

**Generated outputs:**
- `reports/trm_mnist_full_report_*.pdf` - MNIST analysis
- `reports/trm_cifar10_full_report_*.pdf` - CIFAR-10 analysis
- `reports/trm_compare_bounds_report_*.pdf` - Cross-bound comparison
- `plots/` - Individual visualization files

---

## 🏗️ Architecture Overview

```
scripts/trm/
├── core/
│   ├── trm_tiny_train*.py           # Training scripts (MNIST/CIFAR-10)
│   ├── trm_tiny_advtrain*.py        # PGD adversarial training
│   ├── trm_ibp_train*.py            # IBP certified training
│   ├── trm_tiny_verify*.py          # Single-sample verification
│   └── trm_tiny_sweep*.py           # Batch verification sweeps
├── reports/
│   ├── trm_visualize_results.py           # Generate plots
│   ├── trm_full_visual_report_*.py        # PDF reports (MNIST/CIFAR-10)
│   └── trm_compare_bounds_report.py       # Multi-bound comparison
└── presentation/
    └── trm_presentation_slide.py          # PowerPoint generation

checkpoints/          # Trained model weights
logs/                 # CSV verification results
plots/                # Generated visualizations
reports/              # PDF reports
```

---

## 📊 Performance Metrics

**Academic Benchmarks:**
- **MNIST:** ~0.15-0.24s per sample
- **CIFAR-10:** ~0.09-0.24s per sample
- **GPU Memory:** 18-53 MB per sample (A100)

**Production Scale (Beluga):**
- **105.8M parameters:** 2.6s per sample
- **GPU Speedup:** 5× faster than CPU baseline
- **Energy Efficiency:** 4× improvement (reduced time + lower power)

**Bound Method Comparison:**
- **CROWN:** Fastest, baseline accuracy
- **α-CROWN:** +0-5% accuracy, ~1.2× slower
- **β-CROWN:** +0-9% accuracy (baselines only), ~1.5× slower

---

## 🧭 Roadmap

| Stage | Goal | Status |
|--------|------|--------|
| 1️⃣ | CUDA acceleration (A100 verified) | ✅ |
| 2️⃣ | TRM-MLP recursive architecture | ✅ |
| 3️⃣ | Multiple training methods (Baseline, IBP, PGD) | ✅ |
| 4️⃣ | Cross-dataset validation (MNIST + CIFAR-10) | ✅ |
| 5️⃣ | Multi-bound comparison (CROWN, α/β-CROWN) | ✅ |
| 6️⃣ | Comprehensive reporting & visualization | ✅ |
| 7️⃣ | Production scale (105M params, Airbus logistics) | ✅ |
| 8️⃣ | Multi-GPU distributed verification | 🔜 |
| 9️⃣ | Scale to ImageNet & larger models | 🔜 |

---

## 🏆 Hackathon Achievements

**OpenACC Hackathon 2025 - Team Veriphi (TU Wien)**

- 🥇 First-ever verification of 105.8M parameter constraint satisfaction model
- 🚀 5× GPU speedup with attack-guided strategy (85% time reduction)
- 🔬 Research finding: training method effectiveness ↔ dataset complexity
- 📊 Comprehensive cross-dataset validation (MNIST + CIFAR-10)
- ⚡ Successfully profiled with Nsight Systems
- 🌍 Real-world application: Airbus Beluga aerospace logistics

**Team:** Pratik Deshmukh, Vasili Savin, Kartik Arya  
**Mentors:** Vinay Deshpande (Nvidia), Mark Dokter (Know Center)

---

## 📒 Guides

- [VSC5 Connection Guide (CLI)](./docs/vsc5_connection_readme.md)
- [Benchmarking Guide](./docs/trm_scaling_readme.md)
- [Final Presentation](./docs/Presentation.pdf/)

---

## 📚 References

- **auto‑LiRPA Docs:** https://auto-lirpa.readthedocs.io/  
- **α,β‑CROWN Repo:** https://github.com/Verified-Intelligence/alpha-beta-CROWN  
- **Tiny Recursive Models:** https://github.com/SamsungSAILMontreal/TinyRecursiveModels  
- **VNN‑COMP:** https://sites.google.com/view/vnn2024  
- **TUPLES Challenge:** https://arxiv.org/abs/2510.04871

---

## 📄 License

MIT — see `LICENSE`.

---
 
**"Bridging adversarial testing and formal verification for truly robust neural networks."**

*Enabling real-world AI safety validation in safety-critical applications.*
