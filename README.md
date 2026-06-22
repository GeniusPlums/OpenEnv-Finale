# Role Drift Environment

**An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible environment and benchmark for measuring conversational agent drift — with composable programmatic rewards, a GRPO training recipe, and pre-registered held-out evaluation.**

Production voice agents built on small, low-latency language models routinely fail in ways prompts cannot fix: they cannot end calls, abandon assigned tasks, violate explicit instructions, and switch languages mid-conversation. This project turns those failure modes into **composable programmatic detectors**, provides a **reproducible eval harness** with bootstrap confidence intervals, and includes a **GRPO** training pipeline (TRL ecosystem) against a frozen adversarial customer simulator.

**What the V10 benchmark showed:** training return rose on the training distribution (peak group-mean **3.215** at episode 98), but **held-out eval falsified** the hypothesis that this checkpoint beats the prompted baseline. See [BENCHMARK.md](BENCHMARK.md) for verdicts and numbers.

| | |
|---|---|
| **Code** | [github.com/GeniusPlums/OpenEnv-Finale](https://github.com/GeniusPlums/OpenEnv-Finale) |
| **V9 checkpoint (artifact)** | [huggingface.co/GeniusPlums/role-drift-qwen-1-5b-grpo](https://huggingface.co/GeniusPlums/role-drift-qwen-1-5b-grpo) |
| **Eval results (V10)** | [huggingface.co/datasets/GeniusPlums/role-drift-eval-results](https://huggingface.co/datasets/GeniusPlums/role-drift-eval-results) |
| **Docs** | [Index](docs/INDEX.md) · [Reproduction](docs/REPRODUCTION.md) · [Architecture](docs/ARCHITECTURE.md) · [Benchmarks](BENCHMARK.md) |
| **Contribute** | [CONTRIBUTING.md](CONTRIBUTING.md) |

### Five-minute path

1. **Problem** — §1: four drift types in deployable voice agents.
2. **Environment** — §4: OpenEnv `reset` / `step`, frozen 7B customer, scenario JSONL.
3. **Benchmark verdicts** — [BENCHMARK.md](BENCHMARK.md): pre-registered hypotheses, train vs eval metrics, reward-hacking checks.
4. **Reproduce** — [docs/REPRODUCTION.md](docs/REPRODUCTION.md): install, smoke test, eval, train.

---

## 1. Problem

Conversational agents in production—especially **voice agents**—must respond in real time. That imposes a **latency ceiling** (~500ms for the LLM) which forces deployment of **1–7B parameter models**, not frontier systems. These models are fast enough but **drift**:

| Drift type | What goes wrong |
|------------|-----------------|
| **Termination** | Agent cannot exit after the customer says goodbye — thank-you loops |
| **Goal** | Agent abandons its assigned task and becomes a generic assistant |
| **Instruction** | Agent violates explicit prompt rules (fee phrasing, list format, mention limits) |
| **Language** | Agent switches language without user request |

These are **behavioral attractors**, not knowledge gaps. Longer prompts help marginally; they do not provide gradient signal.

---

## 2. What this repository provides

The eval industry (Coval, Hamming, Cekura, Roark) ships **detection** — dashboards of failures handed back to humans for prompt patching. This project standardizes the **measurement and falsification** side of the training loop:

1. **Reproduce** drift under adversarial customer pressure in text (no audio pipeline required).
2. **Score** each failure mode with an independent, inspectable detector.
3. **Run GRPO** against those scores (recipe + checkpoint on Hub — not a validated deployment win).
4. **Evaluate** on held-out scenario IDs with bootstrap CIs and pre-registered hypotheses.

The origin is operational: a services deployment for a client with **500 cold-callers**, where STT/TTS were commodity choices and the **LLM was the hard part**. See [docs/RESEARCH_OVERVIEW.md](docs/RESEARCH_OVERVIEW.md) for the full research narrative.

---

## 3. Real-world examples

Three production failure transcripts ground the environment design and detector validation:

### Masters' Union (admissions) — goal drift

The agent was tasked with recovering an incomplete college application. Mid-call it became a **startup ideation and land-procurement consultant** — despite a long system prompt with dozens of guardrails.

→ `data/transcripts/masters_union_failure.txt` · scenario family `goal_mu_*`

### Kundan Kishore (trading workshop) — termination + language drift

The agent entered a **15+ turn thank-you loop** and switched to **Spanish** unprompted, while also revealing it was an AI.

→ `data/transcripts/kundan_kishore_failure.txt` · scenarios `term_kk_*`, `lang_kk_*`

### DearConnect (broker platform) — instruction drift

The agent dumped **numbered feature lists** despite a one-idea-per-turn rule and mishandled soft refusals.

→ `data/transcripts/dearconnect_failure.txt` · `transfer_dearconnect.jsonl`

Detectors are regression-tested against these transcripts: `tests/test_detectors_on_real_transcripts.py`.

---

## 4. Environment design

An **episode** is one full agent–customer dialogue.

```
reset(scenario_id)  →  load prompt + persona + opening message
step(utterance)     →  score turn reward  →  customer replies  →  until done
```

| Design choice | Rationale |
|---------------|-----------|
| **Frozen customer** | Only the agent learns — avoids self-play instability |
| **LLM customer (7B)** | Adversarial pressure is stochastic and realistic |
| **Text-only** | Voice latency story without STT/TTS complexity |
| **Scenario JSONL** | Reproducible IDs, disjoint train/eval splits |
| **OpenEnv API** | `reset` / `step` / `state` — local or HTTP server |

**Scenario splits**

| File | Scenarios | Use |
|------|-----------|-----|
| `data/scenarios/train.jsonl` | 40 | GRPO training |
| `data/scenarios/eval.jsonl` | 10 | In-domain held-out |
| `data/scenarios/transfer_dearconnect.jsonl` | 8 | Domain shift |
| `data/scenarios/eval_held_out_persona.jsonl` | 4 | Combined-pressure persona |

Full flow diagrams: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 5. Reward architecture

`RewardComposer` sums weighted detector outputs plus anti-gaming bonuses, clipped to **[-5, 5]** per turn.

| Detector | Signal | Implementation |
|----------|--------|----------------|
| `termination_drift` | Talks past customer goodbye | Farewell heuristics + disengagement counter |
| `goal_drift` | Off-topic replies | Embedding similarity to `task_description` (threshold 0.18) |
| `instruction_drift` | Rule violations | Regex rules from `data/prompts/rules/*.json` |
| `language_drift` | Unprompted language switch | `langdetect` vs conversation baseline |

**Bonuses:** `task` (clean on-policy turn), `lang_anchor` (first-turn language match), `pos_lang`, `pos_term` (successful call closure).

**Calibration matters (training distribution only).** An early 20-episode diagnostic *worsened* returns until goal threshold and language detector bugs were fixed; a second diagnostic showed rising training return (slope +0.14 on the **training** reward). That does **not** imply held-out eval improvement — V10 eval falsified H1/H2. See [BENCHMARK.md § Train/eval disconnect](BENCHMARK.md#traineval-disconnect).

Source of truth: `role_drift_env/server/rewards/composer.py`.

---

## 6. Training setup (V9 recipe)

| Component | Choice |
|-----------|--------|
| **Policy** | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| **Customer sim** | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) via vLLM (OpenAI-compatible API) |
| **Algorithm** | GRPO — group-relative advantages, episode-as-sample |
| **Infrastructure** | Hugging Face Jobs (H200), CUDA bootstrap for NVSwitch readiness |
| **V9 config** | 100 episodes · group size 4 · max 6 turns per rollout (GPU memory) |

```bash
# Local entry (GPU) — after vLLM is serving the 7B customer
export ROLE_DRIFT_PERSONA_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
python training/train_grpo.py --episodes 100 --group-size 4 --max-turns 6

# Remote training (recommended)
python scripts/launch_hf_stable_h200_job.py   # after run_stable_h200.sh is on main
```

**Important:** vLLM must be running and the persona gate must pass (non-fallback customer text) before training or eval. See [docs/ARCHITECTURE.md § vLLM](docs/ARCHITECTURE.md#vllm-interaction).

---

## 7. Benchmark results

> **Policy:** Training and eval metrics are reported separately. A rising training return does not imply held-out eval improvement. All numbers trace to JSON in `data/eval_results/` or `episode_log.jsonl` on Hub.

### 7.1 Detector and training-distribution evidence

| Result | Evidence |
|--------|----------|
| Detectors align with production transcript failures | `data/validation/detector_validation.md`, `tests/test_detectors_on_real_transcripts.py` |
| Uncalibrated detectors → negative learning (diag1) | 20 ep training reward slope **−0.20** |
| Calibrated detectors → positive training-return slope (diag2) | 20 ep slope **+0.14**, `plots/diag2/` |

### 7.2 Training metrics (V9 — not eval)

| Metric | Value | Source |
|--------|-------|--------|
| Episodes completed | 100 | `episode_log.jsonl` on Hub |
| Best group-mean training return | **3.215** (episode **98**) | Same log |

These describe **optimization on the training reward**. They are not held-out eval scores.

### 7.3 Held-out evaluation (V10)

| Comparison | Baseline mean [95% CI] | Trained mean [95% CI] | n |
|------------|------------------------|----------------------|---|
| In-domain (`eval.jsonl`) | **0.185** [−0.19, 0.55] | **−1.580** [−1.84, −1.34] | 50 each |
| Transfer (DearConnect) | **0.476** [−0.04, 1.04] | **−0.818** [−0.99, −0.64] | 40 / **15** |

Transfer trained eval is **incomplete** (15 of 40 episodes). Direction on available data: trained below baseline.

### 7.4 Pre-registered hypothesis verdicts

| ID | Claim | Verdict |
|----|-------|---------|
| H1 | Trained > prompted in-domain (non-overlapping 95% CIs) | **FALSIFIED** |
| H2 | Trained > prompted on DearConnect transfer | **FALSIFIED** |
| H3 | Held-out combined-pressure persona | **SKIPPED** |
| H4 | Improvement order: term > instr > goal > lang | **FALSIFIED** |

Reward-hacking (trained vs trivial policies on eval): **FALSIFIED** — trained mean **−1.58** below all four trivial policies (best trivial **−0.07**).

Full tables, per-drift deltas, and reproduction: **[BENCHMARK.md](BENCHMARK.md)** · registration: **[docs/hypotheses.md](docs/hypotheses.md)**

### 7.5 Figures

```bash
hf download GeniusPlums/role-drift-eval-results --local-dir data/eval_results --repo-type dataset
hf download GeniusPlums/role-drift-qwen-1-5b-grpo episode_log.jsonl \
  --local-dir data/training_logs/run_v9
pip install -e ".[plots]"
python scripts/make_plots.py
```

Outputs: `plots/reward_curve.png` (training), `plots/eval_comparison.png`, `plots/transfer.png`, `plots/per_drift_type_curve.png`. Bar charts show trained **below** baseline on synced V10 JSONs.

---

## 8. Reproducing experiments

### 8.1 Install

```bash
git clone https://github.com/GeniusPlums/OpenEnv-Finale.git
cd OpenEnv-Finale
pip install -e .
pip install vllm sentence-transformers langdetect  # GPU path
```

### 8.2 Environment smoke (CPU)

```bash
pytest tests/ -q -k "not grpo"
python -c "
from role_drift_env.server.environment import RoleDriftEnvironment as E
o, s = E().reset('term_kk_01')
print('system_prompt chars:', len(o.system_prompt or ''))
"
```

### 8.3 Start vLLM customer (separate terminal)

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 \
  --gpu-memory-utilization 0.35 --enforce-eager
export ROLE_DRIFT_PERSONA_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
```

### 8.4 Evaluation

```bash
python scripts/run_eval.py in_domain \
  --policy-checkpoint Qwen/Qwen2.5-1.5B-Instruct \
  --output data/eval_results/in_domain_baseline.json

python scripts/run_eval.py in_domain \
  --policy-checkpoint GeniusPlums/role-drift-qwen-1-5b-grpo \
  --output data/eval_results/in_domain_trained.json
```

See `python scripts/run_eval.py --help` for seeds, transfer mode, and Hub upload flags.

### 8.5 Training (GPU)

```bash
python training/train_grpo.py \
  --episodes 100 --group-size 4 --max-turns 6 \
  --output-dir checkpoints/run1
```

### 8.6 Full reproduction guide

[docs/REPRODUCTION.md](docs/REPRODUCTION.md) — canonical install, vLLM, eval, HF Jobs, plots.

---

## 9. Repository structure

```
OpenEnv-Finale/
├── role_drift_env/          # OpenEnv environment package
├── training/                # GRPO trainer + rollout
├── data/scenarios/          # Train / eval / transfer JSONL
├── data/eval_results/       # V10 eval JSON outputs
├── scripts/run_eval.py      # V10 eval harness
├── scripts/make_plots.py    # Figure generation
├── docs/                    # Architecture, research overview, benchmarks
├── BENCHMARK.md             # Verdict table + train/eval tables
└── README.md
```

---

## 10. Future work

| Priority | Item |
|----------|------|
| **Science** | Diagnose train/eval disconnect (rollout length, reward gaming, distribution shift) |
| **Benchmark** | Complete transfer trained eval (15→40 episodes); run H3 persona eval |
| **Detectors** | Stronger goal detector; per-detector training logs |
| **Ops** | Single blessed HF launcher; push `run_stable_h200.sh` + CUDA bootstrap |
| **Community** | Scenario PRs; OpenEnv hub listing as reference falsification environment |

Contributions welcome on **scenarios**, **detectors**, and **eval methodology** — see [docs/REPO_AUDIT.md](docs/REPO_AUDIT.md).

---

## Citation

```bibtex
@software{role_drift_env2026,
  title   = {Role Drift Environment: OpenEnv Benchmark for Conversational Agent Drift},
  author  = {GeniusPlums},
  year    = {2026},
  url     = {https://github.com/GeniusPlums/OpenEnv-Finale}
}
```

---

## License

See repository license file. Production transcripts are redacted where noted (`*.redacted.txt`).
