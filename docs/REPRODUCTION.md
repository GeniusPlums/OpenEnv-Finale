# Reproduction guide

This document is the canonical path to **install**, **validate**, **evaluate**, and **train** against the Role Drift Environment. Every command is runnable from a clean clone unless noted.

**Honesty policy:** Publication headline numbers live in [BENCHMARK.md](../BENCHMARK.md). V10 pre-registered hypotheses H1, H2, H4 were **falsified** on synced Hub JSONs; training and eval metrics must be reported separately.

**Related:** [README](../README.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [scripts/README.md](../scripts/README.md)

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Tested on Linux (HF Jobs) and local dev |
| GPU (training / full eval) | Single GPU with enough VRAM for **7B vLLM customer + 1.5B policy** |
| vLLM | Serves frozen customer at `ROLE_DRIFT_PERSONA_OPENAI_BASE_URL` |
| Hugging Face account | For Hub model download and optional Jobs |

---

## 1. Clone and install

```bash
git clone https://github.com/GeniusPlums/OpenEnv-Finale.git
cd OpenEnv-Finale
pip install -e .
```

Optional extras:

```bash
pip install -e ".[plots]"          # matplotlib for make_plots.py
pip install vllm sentence-transformers langdetect   # GPU rollout path
```

---

## 2. CPU validation (no GPU)

Confirms environment package, detectors, and scenario loading.

```bash
pytest tests/ -q -k "not grpo"
```

Detector regression on production failure transcripts:

```bash
pytest tests/test_detectors_on_real_transcripts.py -v
```

One-line environment smoke:

```bash
python -c "
from role_drift_env.server.environment import RoleDriftEnvironment as E
o, s = E().reset('term_kk_01')
print('scenario:', s.scenario_id)
print('system_prompt chars:', len(o.system_prompt or ''))
print('customer:', (o.customer_message or '')[:80])
"
```

**Expected:** A scenario loads with a non-empty system prompt and customer opening message.

---

## 3. Start the frozen customer (vLLM)

Training and LLM-backed eval require a **7B customer** via vLLM. In a separate terminal:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 \
  --gpu-memory-utilization 0.35 --enforce-eager
export ROLE_DRIFT_PERSONA_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
```

Verify the persona gate (non-fallback customer text):

```bash
python scripts/verify_scripted_personas.py   # scripted path only
# For LLM path: run a short rollout smoke after vLLM is up
```

**Failure mode:** If vLLM is down or OOM, the customer falls back to scripted text — reward dynamics change. Do not compare runs where one used fallback and one did not.

See [ARCHITECTURE.md § vLLM](ARCHITECTURE.md#vllm-interaction).

---

## 4. Evaluation

### 4.1 Eval harness

```bash
python scripts/run_eval.py --help
```

Typical in-domain baseline (prompted policy):

```bash
python scripts/run_eval.py in_domain \
  --policy-checkpoint Qwen/Qwen2.5-1.5B-Instruct \
  --output data/eval_results/in_domain_baseline.json
```

Trained checkpoint:

```bash
python scripts/run_eval.py in_domain \
  --policy-checkpoint path/to/checkpoint \
  --output data/eval_results/in_domain_trained.json
```

Transfer (DearConnect domain shift):

```bash
python scripts/run_eval.py transfer \
  --policy-checkpoint Qwen/Qwen2.5-1.5B-Instruct \
  --output data/eval_results/transfer_baseline.json
```

### 4.2 Compare two JSON outputs

```bash
python scripts/compare_eval_runs.py \
  data/eval_results/in_domain_baseline.json \
  data/eval_results/in_domain_trained.json
```

Prints `mean_total_reward` and 95% bootstrap intervals when the V10 schema includes an `aggregate` block.

### 4.3 Download pre-computed V10 bundle (Hub)

```bash
hf download GeniusPlums/role-drift-eval-results --local-dir data/eval_results
```

Expected filenames for [BENCHMARK.md](../BENCHMARK.md):

| File | Content |
|------|---------|
| `in_domain_baseline.json` | Prompted baseline, eval.jsonl |
| `in_domain_trained.json` | GRPO-trained checkpoint |
| `transfer_baseline.json` | DearConnect transfer, baseline |
| `transfer_trained.json` | DearConnect transfer, trained |
| `reward_hacking_probes.json` | Trivial policy probes |

### 4.4 Local smoke JSONs (not publication)

The repo may contain smaller dev runs with different schemas (`summary` instead of `aggregate`):

| File | What it is |
|------|------------|
| `baseline_sft_indomain.json` | 4 scenarios, 1 seed — SFT-labeled backend |
| `grpo_tiny_indomain.json` | Same subset, tiny GRPO checkpoint |
| `hh_smoke_eval_*.json` | 1–2 scenario pipeline smokes |

Do **not** cite these as V10 headline results. See [BENCHMARK.md § Local smoke](../BENCHMARK.md#local-smoke-runs-not-for-publication).

---

## 5. Training (local GPU)

```bash
export ROLE_DRIFT_PERSONA_OPENAI_BASE_URL=http://127.0.0.1:8000/v1

python training/train_grpo.py \
  --episodes 100 \
  --group-size 4 \
  --max-turns 6 \
  --output-dir checkpoints/run1
```

GRPO smokes (require GPU + vLLM):

```bash
pytest tests/smoke_test_grpo.py -v
```

**CUDA on HF H200:** Use `scripts/wait_for_cuda.py` and the blessed shell entry — see [PROJECT_STATUS.md](PROJECT_STATUS.md). Local `train_grpo.py` may call `bootstrap_cuda.py` before importing torch.

---

## 6. Training (Hugging Face Jobs)

**Recommended path (after `run_stable_h200.sh` is on your branch):**

```bash
python scripts/launch_hf_stable_h200_job.py
```

In-container entry:

```bash
cd /workspace/role-drift-env && bash scripts/run_stable_h200.sh
```

**Legacy path (on `origin/master` today):**

```bash
python scripts/launch_hf_v9_training_job.py
```

See [scripts/README.md](../scripts/README.md) for the full launcher decision tree.

---

## 7. Figures and training logs

### 7.1 Download training log from Hub

```bash
hf download GeniusPlums/role-drift-qwen-1-5b-grpo episode_log.jsonl \
  --local-dir data/training_logs/run_v9
```

### 7.2 Generate plots

```bash
pip install -e ".[plots]"
python scripts/make_plots.py
```

Outputs (when inputs exist):

| Figure | Input |
|--------|-------|
| `plots/reward_curve.png` | `data/training_logs/run_v9/episode_log.jsonl` |
| `plots/eval_comparison.png` | `in_domain_*.json` |
| `plots/transfer.png` | `transfer_*.json` |
| `plots/per_drift_type_curve.png` | episode log + scenario drift tags |

Existing internal diagnostic: `plots/diag2/` (20-episode calibrated detector run — not the V9 100-ep run).

---

## 8. Reproducibility checklist

| Check | Command / artifact |
|-------|-------------------|
| Environment loads | §2 CPU validation |
| Detectors match transcripts | `test_detectors_on_real_transcripts.py` |
| vLLM customer active | Persona gate / no `IS_FALLBACK` in logs |
| Eval scenario IDs disjoint from train | `scripts/check_eval_leakage.py` |
| Headline numbers traceable | JSON in `data/eval_results/` → [BENCHMARK.md](../BENCHMARK.md) |
| Figures match JSON | `make_plots.py` from same commit |

---

## 9. Scenario and data layout

| Path | Scenarios | Use |
|------|-----------|-----|
| `data/scenarios/train.jsonl` | 40 | GRPO training |
| `data/scenarios/eval.jsonl` | 10 | In-domain held-out |
| `data/scenarios/transfer_dearconnect.jsonl` | 8 | Domain shift |
| `data/scenarios/eval_held_out_persona.jsonl` | 4 | Combined-pressure persona |

Add scenarios via PR — see [CONTRIBUTING.md](../CONTRIBUTING.md).

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Negative learning early in training | Miscalibrated detectors | See diag1/diag2 in [RESEARCH_OVERVIEW.md](RESEARCH_OVERVIEW.md) |
| `IS_FALLBACK: True` | vLLM not ready / OOM | Lower `--gpu-memory-utilization`, check port 8000 |
| CUDA error 802 on HF H200 | NVSwitch not ready | `wait_for_cuda.py` must run before torch/vLLM |
| `run_stable_h200.sh: No such file` | Script not on cloned branch | Use V9 launcher or push stable entry |
| `make_plots.py` skips figures | Missing JSON or episode log | Download Hub artifacts (§4.3, §7.1) |

For operational history: [PROJECT_STATUS.md](PROJECT_STATUS.md).
