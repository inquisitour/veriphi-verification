# 🧠 Veriphi: Attack-Guided Neural Network Verification

**GPU-accelerated verification combining attack-guided adversarial search with formal α,β-CROWN certification.**

[![Paper](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org)
[![Models](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/ludwigw)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Research Question:** Is this model provably robust within ε under L∞ perturbations?

Developed at **AI Safety Hackathon 2025** (TU Wien) • Ranked **#3 on Europe's HPC Portal**

---

## 🎯 Key Contributions

**📄 Published Research:** [Veriphi: Attack-Guided Neural Network Verification with Dataset-Dependent Training Methods](https://arxiv.org)

**🔬 Research Finding:** Training method effectiveness is fundamentally dataset-dependent:
- **Simple datasets (MNIST, 784 dim):** IBP training achieves 78% verified accuracy
- **Complex datasets (CIFAR-10, 3072 dim):** PGD adversarial training achieves 94% verified accuracy

**⚡ Attack-Guided Speedup:** 5× faster verification (85% time reduction) by eliminating falsifiable cases with fast attacks before formal verification

**🏭 Production Scale:** First-ever verification of 105.8M parameter model on real Airbus Beluga aerospace logistics (2.6s/sample on A100)

---

## 📊 Main Results

### MNIST (28×28, β-CROWN, 512 samples)

| Training Method | ε=0.04 | ε=0.06 | ε=0.08 | ε=0.1 |
|----------------|--------|--------|--------|-------|
| **IBP (ε=0.01)** | 47% | 61% | **78%** | 75% |
| **PGD (ε=0.15)** | 43% | 48% | 63% | 63% |
| **Baseline** | 0% | 0% | 0% | 0% |

**Winner:** IBP (+15% over PGD at ε=0.08)

### CIFAR-10 (32×32 RGB, β-CROWN, 512 samples)

| Training Method | ε=0.001 | ε=0.002 | ε=0.004 | ε=0.006 | ε=0.008 |
|----------------|---------|---------|---------|---------|---------|
| **PGD (ε=8/255)** | **94%** | **90%** | **80%** | 67% | 58% |
| **IBP (ε=2/255)** | 78% | 51% | 10% | 1% | 0% |
| **Baseline** | 82% | 55% | 13% | 1% | 0% |

**Winner:** PGD (+16% over IBP at ε=0.001, completely dominates at higher ε)

---

## 🚀 Quick Start

```bash
git clone https://github.com/inquisitour/veriphi-verification.git
cd veriphi-verification

python3 -m venv venv
source venv/bin/activate

pip install git+https://github.com/Verified-Intelligence/auto_LiRPA.git
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

export PYTHONPATH="$PWD/src:$PYTHONPATH"
python scripts/core_smoke.py
```

---

## 🧪 Reproduce Paper Results

### Train Models

```bash
# MNIST
python scripts/trm/core/trm_tiny_train.py              # Baseline
python scripts/trm/core/trm_ibp_train.py               # IBP (ε=0.01, 0.15)
python scripts/trm/core/trm_tiny_advtrain.py           # PGD (ε=0.15, 0.20)

# CIFAR-10
python scripts/trm/core/trm_tiny_train_cifar10.py      # Baseline
python scripts/trm/core/trm_ibp_train_cifar10.py       # IBP (ε=2/255)
python scripts/trm/core/trm_tiny_advtrain_cifar10.py   # PGD (ε=8/255)
```

### Run Verification Sweeps

```bash
# MNIST (512 samples, β-CROWN)
python scripts/trm/core/trm_tiny_sweep.py \
  --checkpoint checkpoints/trm_mnist_ibp_eps015_weights.pt \
  --samples 512 \
  --eps 0.01,0.02,0.03,0.04,0.06,0.08,0.1 \
  --bound beta-CROWN

# CIFAR-10 (512 samples, β-CROWN)
python scripts/trm/core/trm_tiny_sweep_cifar10.py \
  --checkpoint checkpoints/trm_cifar10_adv_eps007.pt \
  --samples 512 \
  --eps 0.001,0.002,0.004,0.006,0.008,0.01 \
  --bound beta-CROWN
```

### Generate Reports

```bash
python scripts/trm/reports/trm_full_visual_report_mnist.py
python scripts/trm/reports/trm_full_visual_report_cifar10.py
python scripts/trm/reports/trm_compare_bounds_report.py
```

Outputs: `plots/*.png`, `reports/*.pdf`

---

## 🤗 Pretrained Models

All models available on Hugging Face: [`ludwigw`](https://huggingface.co/ludwigw)

**MNIST:**
- [`trm-mnist-baseline`](https://huggingface.co/ludwigw/trm-mnist-baseline)
- [`trm-mnist-ibp-eps001`](https://huggingface.co/ludwigw/trm-mnist-ibp-eps001) (ε=0.01)
- [`trm-mnist-ibp-eps015`](https://huggingface.co/ludwigw/trm-mnist-ibp-eps015) (ε=0.15) ⭐
- [`trm-mnist-adv-eps015`](https://huggingface.co/ludwigw/trm-mnist-adv-eps015) (PGD ε=0.15)
- [`trm-mnist-adv-eps020`](https://huggingface.co/ludwigw/trm-mnist-adv-eps020) (PGD ε=0.20)

**CIFAR-10:**
- [`trm-cifar10-pgd`](https://huggingface.co/ludwigw/trm-cifar10-pgd) (PGD ε=8/255) ⭐

**Beluga (105.8M):**
- [`beluga-trm-105m`](https://huggingface.co/ludwigw/beluga-trm-105m) (Airbus logistics)

---

## 🏭 Production: Airbus Beluga Logistics

**Dataset:** 2,336 real Airbus Beluga XL constraint satisfaction problems
- Variable dimensions: 69-821 jigs, 43-199 flights per problem
- 4 constraint types: flight capacity, rack capacity, scheduling, type matching

**Performance:**
- **Training loss:** 930 → 2.26 (99.8% reduction)
- **Verification:** 2.6s per sample (A100)
- **Model:** 105.8M parameters with dynamic padding/masking

```bash
# Train
python scripts/beluga/train_beluga.py --epochs 50 --device cuda

# Verify
python scripts/beluga/verify_beluga_sweep.py \
  --checkpoint checkpoints/beluga_trm_105M.pt \
  --samples 50
```

---

## 📈 Performance Metrics

**Verification Time (A100):**
- MNIST: 0.15-0.24s per sample
- CIFAR-10: 0.09-0.24s per sample  
- Beluga (105.8M): 2.6s per sample

**GPU Memory:**
- Academic benchmarks: 18-53 MB per sample
- Production (105.8M): Efficient scaling validated

**Bound Method Comparison:**
- **CROWN:** Baseline (fastest)
- **α-CROWN:** +0-5% accuracy, ~1.2× slower
- **β-CROWN:** +0-9% accuracy, ~1.5× slower

---

## 🏗️ Architecture

```
scripts/trm/
├── core/                  # Training & verification
│   ├── trm_tiny_train*.py
│   ├── trm_ibp_train*.py
│   ├── trm_tiny_advtrain*.py
│   └── trm_tiny_sweep*.py
├── reports/               # Visualization & PDFs
└── presentation/          # PowerPoint generation

checkpoints/               # Model weights (.pt)
logs/                      # CSV verification results
plots/                     # Generated figures
reports/                   # PDF reports
```

---

## 🎓 Citation

```bibtex
@article{deshmukh2026veriphi,
  title={Veriphi: Attack-Guided Neural Network Verification with Dataset-Dependent Training Methods},
  author={Deshmukh, Pratik and Savin, Vasili and Arya, Kartik},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## 🏆 Team & Recognition

**Team Veriphi** (TU Wien):
- Pratik Deshmukh
- Vasili Savin
- Kartik Arya

**Mentors:**
- Vinay Deshpande (Nvidia)
- Mark Dokter (Know Center)

**Recognition:**
- 🥉 **#3 on Europe's HPC Portal** - "Ten Projects that Boosted AI Performance with GPUs"
- 🏅 AI Safety Hackathon 2025, TU Wien

---

## 📚 References

- [auto-LiRPA Documentation](https://auto-lirpa.readthedocs.io/)
- [α,β-CROWN Repository](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
- [Tiny Recursive Models](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- [VNN-COMP](https://sites.google.com/view/vnn2024)

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

**"Bridging adversarial testing and formal verification for production AI safety."**
