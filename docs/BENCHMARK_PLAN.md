# Benchmark Publication Plan

**Date:** 2026-06-22  
**Status:** Planning document — no results invented here  
**Related:** [BENCHMARK.md](../BENCHMARK.md), [RESEARCH_OVERVIEW.md](RESEARCH_OVERVIEW.md)

---

## Current state (facts)

### What exists

| Asset | Location | Notes |
|-------|----------|-------|
| Scenario splits | `data/scenarios/` | 40 train / 10 eval / 8 transfer / 4 held-out / 4 injection |
| Eval harness | `scripts/run_eval.py` | Bootstrap CIs, `by_drift_type`, Hub upload |
| Plot generator | `scripts/make_plots.py` | 4 figure types; skips missing inputs |
| Pre-registered hypotheses | `docs/hypotheses.md` | H1–H4 |
| Hub model (reported V9) | [GeniusPlums/role-drift-qwen-1-5b-grpo](https://huggingface.co/GeniusPlums/role-drift-qwen-1-5b-grpo) | README cites `episode_log.jsonl`, best ~3.215 group mean |
| Hub eval dataset (target) | [GeniusPlums/role-drift-eval-results](https://huggingface.co/datasets/GeniusPlums/role-drift-eval-results) | Five JSON files per BENCHMARK.md |
| Diag2 learning signal | `plots/diag2/`, `docs/hackathon_progress.md` | 20 ep, slope +0.14 — internal validation only |
| Local smoke evals | `data/eval_results/baseline_sft_*`, `grpo_tiny_*` | 4 scenarios, 1 seed — **not** publication tier |

### What is missing

| Gap | Impact |
|-----|--------|
| V10 Bundle A JSONs not in local `data/eval_results/` | Cannot fill BENCHMARK.md or generate README figures |
| `run_v9/episode_log.jsonl` not in repo | Training curve figures require Hub download |
| `BENCHMARK.md` headline table all TBD | Public README cites plots that do not exist locally |
| `reward_hacking_probe.json` incomplete (`trained: null`) | Hacking probe section cannot be finalized |
| H3 held-out persona eval skipped | Pre-registered hypothesis unresolved |
| No multi-seed training runs | Cannot claim training stability |
| No per-detector learning curves from training | Only aggregate `mean_return` logged |
| No latency / cost metrics | Weak production-deployment story |

---

## Target: publication-quality benchmark package

A "publication-quality" release for this project means a reader can:

1. Reproduce scenario IDs and detector code from the repo
2. Download frozen eval JSONs and verify every number in `BENCHMARK.md`
3. Regenerate all figures with one command
4. See honest limitations and skipped hypotheses

This is workshop-paper / technical-report tier, not a new SOTA claim on general chat benchmarks.

---

## Phase 1 — Artifact recovery (1–2 days)

### 1.1 Download canonical artifacts from Hub

```bash
# Eval results (Bundle A)
hf download GeniusPlums/role-drift-eval-results --local-dir data/eval_results

# Training log from model repo
hf download GeniusPlums/role-drift-qwen-1-5b-grpo episode_log.jsonl \
  --local-dir data/training_logs/run_v9
```

**Verify files present:**

- [ ] `in_domain_baseline.json`
- [ ] `in_domain_trained.json`
- [ ] `transfer_baseline.json`
- [ ] `transfer_trained.json`
- [ ] `reward_hacking_probes.json`
- [ ] `data/training_logs/run_v9/episode_log.jsonl`

### 1.2 Normalize local naming

Rename or symlink smoke files to `data/eval_results/smoke/` so V10 files are unambiguous.

### 1.3 Fill BENCHMARK.md from JSON only

For each headline metric, copy:

- `aggregate.mean_total_reward`
- `aggregate.ci_95_low`, `aggregate.ci_95_high` (if present)
- `by_drift_type` blocks

**Rule:** If a JSON field is absent, write "not reported" — do not estimate.

---

## Phase 2 — Core tables & figures (2–3 days)

### 2.1 Required figures (`make_plots.py`)

| Figure | Input | Purpose |
|--------|-------|---------|
| `reward_curve.png` | `episode_log.jsonl` | Training convergence |
| `per_drift_type_curve.png` | episode log + scenario drift map | Which drift types drive learning |
| `eval_comparison.png` | in_domain baseline vs trained | H1 primary result |
| `transfer.png` | transfer baseline vs trained | H2 generalization |

**Enhancement to `make_plots.py`:**

- [ ] Error bars from bootstrap CIs in eval JSON (not just bar height)
- [ ] Per-drift-type grouped bars with CI whiskers
- [ ] Waterfall or stacked component chart (term / goal / instr / lang contributions)

### 2.2 Required tables (for README + BENCHMARK.md)

**Table 1 — In-domain aggregate (H1)**

| Policy | Mean return | 95% CI | N episodes |
|--------|-------------|--------|------------|
| Prompted baseline | from JSON | from JSON | 5 seeds × 10 scenarios |
| GRPO trained | from JSON | from JSON | same |

**Table 2 — Transfer aggregate (H2)**

Same structure for DearConnect 8 scenarios.

**Table 3 — Per drift type (H4)**

| Drift type | Baseline mean | Trained mean | Δ | CI overlap? |
|------------|---------------|--------------|---|-------------|
| termination | | | | |
| goal | | | | |
| instruction | | | | |
| language | | | | |

**Table 4 — Reward hacking probes**

| Trivial policy | Mean return | Beaten by trained? |
|----------------|-------------|-------------------|
| always_empty | | |
| always_rephrase | | |
| always_summary | | |
| mute_after_farewell | | |

**Table 5 — Training summary**

| Metric | Value | Source |
|--------|-------|--------|
| Episodes | 100 | V9 config |
| Best group mean return | ~3.215 | episode_log — verify episode index |
| Final rolling-10 mean | TBD | episode_log |
| Mean KL | TBD | episode_log |

### 2.3 Missing metrics to add to eval harness

| Metric | Implementation | Priority |
|--------|----------------|----------|
| `mean_turns` per policy | Already in episode records — aggregate in summary | P1 |
| `fallback_rate` | Log `IS_FALLBACK` from persona in rollout transcript | P1 |
| `terminal_success_rate` | Aggregate `terminal_success` from rollout | P1 |
| `per_detector_mean` in training log | Extend `train_grpo_core.py` logging | P2 |
| `tokens_generated` per turn | Tokenizer count in rollout | P2 |
| `wall_time_per_episode` | Already optional `--time-each-episode` | P2 |

---

## Phase 3 — Statistical rigor (3–5 days)

### 3.1 Clarify what is bootstrapped vs not

| Claim | Valid with current design? | Upgrade |
|-------|---------------------------|---------|
| Trained > baseline on eval (H1) | Yes — 5 inference seeds × 10 scenarios | Add paired bootstrap per scenario |
| Training improved over time | Single training seed | Run 3 seeds × 100 ep (expensive) |
| Transfer generalization | Same as H1 on 8 scenarios | Report effect size + CI overlap honestly |

### 3.2 Report effect sizes

Add to BENCHMARK.md:

- Absolute Δ mean return (trained − baseline)
- Relative improvement %
- Whether 95% CIs overlap (conservative test)
- Optional: paired permutation test per scenario_id

### 3.3 Resolve pre-registered hypotheses

| ID | Action if Bundle A complete | Action if incomplete |
|----|----------------------------|----------------------|
| H1 | HELD / FALSIFIED / INCONCLUSIVE from CIs | INCONCLUSIVE — state missing JSON |
| H2 | Same on transfer JSONs | INCONCLUSIVE |
| H3 | Mark SKIPPED with reason (documented in V10) | Re-run when compute available |
| H4 | Rank `by_drift_type` deltas | Compare to predicted order; note deviations |

---

## Phase 4 — Baseline completeness (1 week)

### 4.1 Baselines to report

| Baseline | Status | Notes |
|----------|--------|-------|
| Prompted Qwen2.5-1.5B (no GRPO) | In V10 harness | Primary comparison |
| SFT on demonstration data | `train_sft.py` exists; partial eval JSONs | Finish or drop from claims |
| Frontier prompted (GPT-4o / Groq) | `eval_baseline.py` supports Groq | Optional — latency story only |
| Deployable-class prompted 7B | Not implemented | Stretch — same latency class |

### 4.2 Ablations (future work, high value)

| Ablation | Purpose |
|----------|---------|
| Detector off (one at a time) | Which signal drives learning |
| Scripted vs LLM customer | Persona necessity |
| Prompt-only vs GRPO, same turns cap | Isolate RL benefit |
| Threshold sensitivity (goal 0.18) | Detector robustness |

---

## Phase 5 — Reproducibility bundle (ongoing)

Ship a `reproduce/` or documented shell sequence:

```bash
# 1. Environment smoke (CPU)
pytest tests/ -q -k "not grpo"

# 2. Download artifacts
hf download GeniusPlums/role-drift-eval-results --local-dir data/eval_results
hf download GeniusPlums/role-drift-qwen-1-5b-grpo episode_log.jsonl \
  --local-dir data/training_logs/run_v9

# 3. Regenerate figures
pip install -e ".[plots]"
python scripts/make_plots.py

# 4. Optional: re-run eval (GPU + vLLM)
bash scripts/run_v10_eval_worker.sh  # see docs/ops/
```

Add CI job that:

- Runs detector unit tests
- Validates eval JSON schema against a pydantic model
- Fails if README references plots newer than committed checksums (optional)

---

## Chart style guide (publication quality)

| Element | Standard |
|---------|----------|
| Font | Matplotlib default → upgrade to 11pt sans |
| Colors | Colorblind-safe palette (tab10 with patterns) |
| Error bars | 95% bootstrap CI, labeled in caption |
| Sample size | In subtitle: "n = 50 episodes (5 seeds × 10 scenarios)" |
| Version pin | Footer: commit SHA + detector version |
| Negative rewards | Do not truncate y-axis to exaggerate gains |

---

## Timeline estimate

| Phase | Effort | Blocker |
|-------|--------|---------|
| 1 — Artifact recovery | 1–2 days | Hub access, eval job completion |
| 2 — Tables & figures | 2–3 days | Phase 1 |
| 3 — Statistics | 3–5 days | Phase 2 |
| 4 — Baselines | 1 week | GPU compute |
| 5 — Repro bundle | Ongoing | Launcher consolidation |

---

## Honest headline (updated 2026-06-22)

**Verified:** V10 Hub JSONs synced. Pre-registered **H1, H2, H4 falsified**; **H3 skipped**; reward-hacking **falsified**. Training peak **3.215** (ep 98) does not transfer to held-out eval (trained **−1.58** vs baseline **+0.19** in-domain).

**See:** [BENCHMARK.md](../BENCHMARK.md) for verdict table and train/eval disconnect.
