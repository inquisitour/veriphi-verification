# Veriphi: Neural Network Robustness Verification

A compact verification stack that combines **fast adversarial attacks** with **formal bounds (α,β‑CROWN via auto‑LiRPA)**. It answers a simple question:

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
pip install -r requirements.txt -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu121
# CPU-only alt:
# pip install -r requirements.txt -c constraints.txt
```

### (Optional) auto‑LiRPA from source (pinned)
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

Minimal one‑liner check:
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

## 🧪 Tests

```bash
# All tests (unit + integration + benchmarks)
python -m pytest -q

# Or target a suite
python -m pytest tests/unit    -q
python -m pytest tests/integration -q
python -m pytest tests/benchmarks  -q
```

---

## CPU baselines

We keep results under `data/baselines/cpu/` and summaries under `data/baselines/cpu/summary/`.

### Generate baselines
```bash
# From repo root
source venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python scripts/run_cpu_baselines.py
```

That will create a file like:
```
data/baselines/cpu/cpu_baselines_<timestamp>.csv
```

### Summarize baselines
```bash
python scripts/summarize_baselines.py
```

This reads all `data/baselines/cpu/*.csv` and writes grouped summaries to:
```
data/baselines/cpu/summary/summary_cpu_baselines_<timestamp>.csv
```

Each row includes:
- `model, norm, epsilon, verification_rate, runs, avg_time_s, avg_mem_mb`

---

## 🏗️ Architecture

```
src/core/
├── verification/
│   ├── base.py              # Verification interfaces (VerificationEngine, configs, results)
│   ├── alpha_beta_crown.py  # α,β‑CROWN via auto‑LiRPA
│   └── attack_guided.py     # Attack‑guided strategy (attacks → formal)
├── attacks/
│   ├── base.py              # Attack interfaces + registry
│   └── fgsm.py              # FGSM + Iterative FGSM
├── models/
│   └── test_models.py       # Tiny/Linear/Conv test models + factories
└── __init__.py              # VeriphiCore façade (create_core_system, helpers)
```

Key ideas:
- **Attack‑guided**: Try FGSM/I‑FGSM first for fast falsification; if none succeed, run α,β‑CROWN.
- **Deterministic**: Seeds + simple toy models for quick iterations.
- **Extensible**: Add attacks via the registry; add verifiers by implementing the base interface.

---

## 🖥️ CLI & scripts

- `scripts/core_smoke.py` — verifies imports, simple bounds, and engine contracts.
- `scripts/attack_guided_demo.py` — shows attack‑guided flow with logging.
- `scripts/run_cpu_baselines.py` — runs models × norms × ε and writes CSV to `data/baselines/cpu/`.
- `scripts/summarize_baselines.py` — aggregates all CSVs into grouped summaries under `data/baselines/cpu/summary/`.

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

1) **GPU lift** (Step 4): move tensors/models to CUDA, batch inputs, add AMP around bounds for speed.
2) **Models that matter** (Step 5): add stubs for ResNet‑18 (CIFAR‑10) and ResNet‑50 (ImageNet).
3) **Demo scaffolding** (Step 6): minimal web UI — *upload → pick model → ε/norm → verify → report verdict/time/mem*.

---

## 📚 References

- **auto‑LiRPA docs**: https://auto-lirpa.readthedocs.io/
- **α,β‑CROWN repo**: https://github.com/Verified-Intelligence/alpha-beta-CROWN
- **VNN‑COMP**: https://sites.google.com/view/vnn2024

---

## 📄 License

MIT — see `LICENSE`.
