# 🧠 Veriphi: Neural Network Robustness Verification

A **GPU‑accelerated verification stack** combining **attack‑guided adversarial search** with **formal bound certification**  
(α‑, β‑CROWN via [auto‑LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA)).

It answers a simple but critical question:

> **“Is this model provably robust within ε under L∞ or L2 perturbations?”**

…and returns **verified / falsified**, with measured **runtime & memory**.

---

## 🚀 New Highlights

✅ **Attack‑Guided Verification:**  
   Fast falsification via FGSM + I‑FGSM, then formal verification using α‑, β‑CROWN.

✅ **TRM‑MLP Integration:**  
   Support for **Tiny Recursive Models (TRM)** — verified using the same unified pipeline.

✅ **GPU‑Accelerated Verification:**  
   Works seamlessly on **A100, RTX** or any CUDA‑enabled GPU.

✅ **Bound Comparison (CROWN vs α‑, β‑CROWN):**  
   Demonstrates verified fraction improvements through **adversarial training** and **tight bounds**.

---

## 🔧 Install (reproducible)

```bash
# Clone
git clone https://github.com/inquisitour/veriphi-verification.git
cd veriphi-verification

# Python env
python3 -m venv venv
source venv/bin/activate

# Install (use CUDA wheels if you have GPU)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

(Optional) install auto‑LiRPA from source:
```bash
git clone https://github.com/Verified-Intelligence/auto_LiRPA.git
cd auto_LiRPA && git checkout v0.6.0
pip install -e .
cd ..
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

Minimal one-liner check:
```bash
python - <<'PY'
from core import create_core_system
from core.models import create_test_model, create_sample_input
core  = create_core_system(use_attacks=True, device='cpu')
model = create_test_model('tiny'); x = create_sample_input('tiny')
res   = core.verify_robustness(model, x, epsilon=0.1, norm='inf', timeout=30)
print(res.status.value, res.verified, f"{res.verification_time:.3f}s", "mem=", res.additional_info.get("memory_usage_mb"))
PY
```

---

## ⚡ GPU mode

Veriphi now fully supports **CUDA (A100, RTX, etc.)**.

```bash
# Enable GPU device
export VERIPHI_DEVICE=cuda
```

All engines, attacks, and models will automatically run on the GPU.

Check GPU availability:
```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}:", torch.cuda.get_device_name(i))
PY
```

Run a GPU smoke test:
```bash
python scripts/gpu_smoke.py
```

Expected:  
```
CUDA available: True
✓ Attack-guided verification engine initialized on cuda
Verification result: verified ...
```

---

## 🧪 Tests

```bash
# All tests (unit + integration + benchmarks)
export VERIPHI_DEVICE=cuda
python -m pytest -q
```

Or target a suite:
```bash
pytest tests/unit -q
pytest tests/integration -q
pytest tests/benchmarks -q
```

To run the full verification validation:
```bash
export VERIPHI_DEVICE=cuda && python run_tests.py --all --fix-tests
```

---

## 📊 Baselines

We keep results under:
- `data/baselines/cpu/` — CPU performance
- `data/baselines/gpu/` — GPU performance (A100, RTX, etc.)

### Generate CPU baselines
```bash
source venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python scripts/run_cpu_baselines.py
```

### Generate GPU baselines
```bash
source venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export VERIPHI_DEVICE=cuda
python scripts/run_gpu_baselines.py
```

Each run creates:
```
data/baselines/{cpu|gpu}/{cpu|gpu}_baselines_<timestamp>.csv
```

### Summarize baselines
```bash
python scripts/summarize_baselines.py
```

Writes grouped summaries to:
```
data/baselines/{cpu|gpu}/summary/summary_<timestamp>.csv
```

---

## 🧩 TRM Experiments

### Train TRM‑MLP on MNIST
```bash
python scripts/trm_tiny_train.py
```

### Adversarially Fine‑Tune
```bash
python scripts/trm_tiny_advtrain.py
```

### Verify Robustness (attack + formal)
```bash
python scripts/trm_tiny_verify.py
```

Outputs detailed logs for each ε and produces:
```
reports/trm_robustness_report.pdf
```

### Bound Comparison Sweep
```bash
python scripts/trm_tiny_sweep.py
```

Generates cross‑method comparison:
- α‑CROWN
- β‑CROWN
- CROWN baseline

### Visual Report
```bash
python scripts/trm_visualize_results.py
```

Produces:
```
reports/trm_full_visual_report.pdf
```

### Streamlit UI
```bash
chmod +x run_streamlit_safe.sh
```
```bash
./run_streamlit_safe.sh
```

---

## 📊 Example Verified Fractions (TRM, ε = 0.03, L∞)

| Bound Method | Avg Verified Fraction |
|---------------|----------------------:|
| CROWN         | 0.111 |
| α‑CROWN       | 0.143 |
| **β‑CROWN**   | **0.146 ✅** |

---

## 🏗️ Architecture Overview

```
src/core/
├── verification/
│   ├── base.py              # Verification interfaces + configs
│   ├── alpha_beta_crown.py  # α,β‑CROWN formal bound engines
│   └── attack_guided.py     # Orchestrates attacks + verifier
├── attacks/
│   ├── base.py              # Attack interfaces
│   └── fgsm.py              # FGSM + iterative FGSM
├── models/
│   ├── test_models.py       # Tiny/Linear/Conv models
│   ├── resnet_stubs.py      # ResNet‑18/50 demo integration
│   └── trm_adapter.py       # TRM‑MLP + recursive model adapter
└── __init__.py              # VeriphiCore façade
```

---

## 📈 Results Summary

```
✅ TRM Adversarially Trained Model
ε = 0.03, norm = L∞
verified = 7/10
falsified = 3/10
β‑CROWN > α‑CROWN > CROWN
```

Generated visual reports:
- `trm_robustness_report.pdf`
- `trm_compare_bounds_report.pdf`
- `trm_full_visual_report.pdf`
- `trm_hackathon_slide.pptx`
- `trm_hackathon_slide.pdf`

---

## 🧭 Roadmap

| Stage | Goal | Status |
|--------|------|--------|
| 1️⃣ | CUDA acceleration (A100 verified) | ✅ |
| 2️⃣ | Add TRM MLP recursive architecture support | ✅ |
| 3️⃣ | Adversarial + verified robustness training | ✅ |
| 4️⃣ | Visual + PowerPoint auto‑reporting | ✅ |
| 5️⃣ | Heavy runs for 7M parameter TRM models | 🔜 |

---

## 📒 Guides

- # [VSC5 Connection Guide (CLI)](./docs/vsc5_connection_readme.md)
- # [Benchmarking Guide](./docs/trm_scaling_readme.md)

---

## 📚 References

- **auto‑LiRPA Docs:** https://auto-lirpa.readthedocs.io/  
- **α,β‑CROWN Repo:** https://github.com/Verified-Intelligence/alpha-beta-CROWN  
- **Tiny Recursive Models:** https://github.com/SamsungSAILMontreal/TinyRecursiveModels  
- **VNN‑COMP:** https://sites.google.com/view/vnn2024  

---

## 📄 License

MIT — see `LICENSE`.

---
 
“*Bridging adversarial testing and formal verification for truly robust neural networks.*”
