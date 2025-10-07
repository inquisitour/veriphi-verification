# Veriphi: Neural Network Robustness Verification

A compact verification stack that combines **fast adversarial attacks** with **formal bounds (α,β-CROWN via auto-LiRPA)**. It answers a simple question:

> **“Is this model robust within ε under L∞/L2?”**

…and returns **verified / falsified**, with **runtime & memory**. 

---

## 🔧 Install (reproducible)

```bash
# Clone
git clone https://github.com/inquisitour/veriphi-verification.git
cd veriphi-verification

# Python env
python3 -m venv venv
source venv/bin/activate

# Install (uses pinned constraints; add the extra index for CUDA wheels if you have an NVIDIA GPU)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
# CPU-only alt:
# pip install -r requirements.txt
```

### (Optional) auto-LiRPA from source (pinned)
```bash
git clone https://github.com/Verified-Intelligence/auto_LiRPA.git
cd auto_LiRPA
git checkout v0.6.0
pip install -e .
cd ..
```

### Verify your toolchain
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

A “6/6 tests passed” summary indicates the core stack is healthy.

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

## 🏗️ Architecture

```
src/core/
├── verification/
│   ├── base.py              # Verification interfaces (VerificationEngine, configs, results)
│   ├── alpha_beta_crown.py  # α,β-CROWN via auto-LiRPA (GPU-aware)
│   └── attack_guided.py     # Attack-guided strategy (attacks → formal)
├── attacks/
│   ├── base.py              # Attack interfaces + registry
│   └── fgsm.py              # FGSM + Iterative FGSM
├── models/
│   └── test_models.py       # Tiny/Linear/Conv test models + factories (device-aware)
└── __init__.py              # VeriphiCore façade (create_core_system, helpers)
```

Key ideas:
- **Attack-guided**: Try FGSM/I-FGSM first for fast falsification; if none succeed, run α,β-CROWN.
- **Device-aware**: Controlled globally via `VERIPHI_DEVICE` (`cpu` or `cuda`).
- **Deterministic**: Seeds + simple toy models for quick iterations.
- **Extensible**: Add attacks via the registry; add verifiers by implementing the base interface.

---

## 🖥️ CLI & scripts

- `scripts/core_smoke.py` — verifies imports, simple bounds, and engine contracts.
- `scripts/attack_guided_demo.py` — shows attack-guided flow with logging.
- `scripts/run_cpu_baselines.py` — runs CPU baselines.
- `scripts/run_gpu_baselines.py` — identical GPU variant.
- `scripts/resnet_smoke.py` — sanity-checks ResNet-18/50 with attack-guided verifier.
- `scripts/summarize_baselines.py` — aggregates all CSVs into grouped summaries.

---

## 📈 Example output (summary)

```
 model  norm  epsilon  verification_rate  runs  avg_time_s  avg_mem_mb
  tiny   inf    0.050              1.000     1       0.022       439.1
  tiny     2    0.100              1.000     1       0.024       439.8
linear   inf    0.050              0.000     1       0.003       440.0
  conv   inf    0.100              0.000     1       0.003       449.3
```

---

## 🧭 Roadmap (hackathon)

1) ✅ **GPU lift**: full CUDA support with A100 acceleration.
2) ✅ **Models that matter**: added ResNet-18/50 support stubs.
3) 🛠️ **Demo scaffolding**: minimal web UI — *upload → pick model → ε/norm → verify → report verdict/time/mem*.

---

## 📚 References

- **auto-LiRPA docs**: https://auto-lirpa.readthedocs.io/
- **α,β-CROWN repo**: https://github.com/Verified-Intelligence/alpha-beta-CROWN
- **VNN-COMP**: https://sites.google.com/view/vnn2024

---

## 📄 License

MIT — see `LICENSE`.
