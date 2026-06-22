# Benchmark results

**Role Drift Environment — V10 evaluation bundle, training log, and final hypothesis verdicts.**

Every number in this file traces to a JSON artifact in `data/eval_results/` or `data/training_logs/run_v9/episode_log.jsonl` (synced from Hugging Face Hub, 2026-06-22). We do not invent metrics.

| | |
|---|---|
| **Methodology** | [docs/BENCHMARK_PLAN.md](docs/BENCHMARK_PLAN.md) |
| **Reproduce** | [docs/REPRODUCTION.md](docs/REPRODUCTION.md) |
| **Registration** | [docs/hypotheses.md](docs/hypotheses.md) |
| **Hub dataset** | [GeniusPlums/role-drift-eval-results](https://huggingface.co/datasets/GeniusPlums/role-drift-eval-results) |
| **Hub model (V9 log)** | [GeniusPlums/role-drift-qwen-1-5b-grpo](https://huggingface.co/GeniusPlums/role-drift-qwen-1-5b-grpo) |

---

## Final hypothesis verdict table

| ID | Pre-registered claim | Verdict | Evidence (one line) |
|----|----------------------|---------|---------------------|
| **H1** | GRPO-trained 1.5B beats prompted baseline in-domain (non-overlapping 95% CIs) | **FALSIFIED** | Trained **−1.580** [−1.84, −1.34] vs baseline **0.185** [−0.19, 0.55]; n=50; CIs do not overlap |
| **H2** | Trained beats baseline on DearConnect transfer | **FALSIFIED** | Trained **−0.818** [−0.99, −0.64] vs baseline **0.476** [−0.04, 1.04]; trained n=**15**/40 |
| **H3** | Trained beats baseline on combined-pressure held-out persona | **SKIPPED** | No V10 paired baseline/trained eval on `eval_held_out_persona.jsonl` |
| **H4** | Per-drift improvement order: term > instr > goal > lang | **FALSIFIED** | In-domain Δ: term −3.60, instr −2.84, goal −1.62; transfer Δ (partial): lang **+1.51** > goal −2.08 |
| **RH** | Trained beats all four trivial reward-hacking policies | **FALSIFIED** | Trained eval **−1.58**; best trivial (`always_summary`) **−0.07** |

---

## Training metrics (V9)

Source: `data/training_logs/run_v9/episode_log.jsonl` (from Hub model repo).

| Metric | Value | Notes |
|--------|-------|-------|
| Episodes | 100 | GRPO, group size 4, max 6 turns per rollout |
| Best group-mean `mean_return` | **3.2151** | Episode **98** |
| Diagnostic diag2 training-return slope | **+0.14** | 20-ep run after detector calibration; `plots/diag2/` |
| Diagnostic diag1 training-return slope | **−0.20** | 20-ep run before calibration |

**Interpretation:** The policy optimized the **training reward**. These numbers must not be quoted as held-out eval performance.

---

## Evaluation metrics (V10)

Source: `data/eval_results/in_domain_*.json`, `transfer_*.json`. Harness: `scripts/run_eval.py`. Bootstrap 95% CIs on episode returns.

### In-domain (`eval.jsonl`, 10 scenarios × 5 seeds = 50)

| Policy | `mean_total_reward` | 95% CI | n |
|--------|---------------------|--------|---|
| Prompted baseline (`Qwen2.5-1.5B-Instruct`) | **0.1846** | [−0.1871, 0.5520] | 50 |
| GRPO trained (V9 checkpoint) | **−1.5796** | [−1.8421, −1.3396] | 50 |

```bash
python scripts/compare_eval_runs.py \
  data/eval_results/in_domain_baseline.json \
  data/eval_results/in_domain_trained.json
```

### Transfer (`transfer_dearconnect.jsonl`)

| Policy | `mean_total_reward` | 95% CI | n | Expected n |
|--------|---------------------|--------|---|------------|
| Prompted baseline | **0.4755** | [−0.0447, 1.0354] | 40 | 40 |
| GRPO trained | **−0.8179** | [−0.9867, −0.6416] | **15** | 40 |

Transfer trained eval is **incomplete**. Verdict on available data: trained below baseline.

### Per-drift Δ (trained − baseline, aggregate `by_drift_type`)

**In-domain (n=50 each):**

| Drift type | Δ mean reward |
|------------|---------------|
| cooperative | −0.997 |
| goal | −1.623 |
| instruction | −2.844 |
| termination | −3.596 |
| language | *not in aggregate* |

**Transfer (baseline n=40, trained n=15):**

| Drift type | Δ mean reward |
|------------|---------------|
| termination | −0.251 |
| instruction | −2.918 |
| goal | −2.076 |
| language | **+1.505** |

---

## Train/eval disconnect

The V9 run exhibits a clear **train/eval gap**:

| Surface | Best / headline number | What it measures |
|---------|------------------------|------------------|
| Training log | **+3.215** group-mean return (ep 98) | GRPO optimization on training scenarios, 6-turn rollouts |
| Held-out in-domain eval | **−1.580** vs baseline **+0.185** | 10 held-out scenarios, up to 30 turns, prompted vs checkpoint |

Plausible contributors (not isolated in this run):

- **Rollout length mismatch** — train max 6 turns, eval up to 30.
- **Reward gaming** — training return rises while trivial eval policies outscore the checkpoint (see § Reward-hacking).
- **Distribution shift** — train scenario mix vs held-out eval IDs.
- **Detector–eval coupling** — same composer used for train and eval, but policy may exploit training-time structure.

**Primary empirical lesson:** detector calibration is necessary for learnable training signal (diag1 → diag2), but **not sufficient** for held-out improvement under this recipe.

---

## Reward-hacking probes

Source: `data/eval_results/reward_hacking_probes.json` (eval.jsonl, 5 seeds). Trained reference: `in_domain_trained.json` aggregate (**−1.5796**); `trained_reference_mean_return` field in probes JSON is `null`.

| Trivial policy | Mean total reward | 95% CI | Trained beats? |
|----------------|-------------------|--------|----------------|
| `always_empty` | −0.2500 | [−0.30, −0.20] | No |
| `always_rephrase` | −1.2548 | [−1.31, −1.20] | No |
| `always_summary` | −0.0669 | [−0.75, 0.61] | No |
| `mute_after_farewell` | −0.2940 | [−0.45, −0.14] | No |

---

## Supporting evidence (not headline eval)

| Result | Evidence |
|--------|----------|
| Detectors vs production transcripts | `tests/test_detectors_on_real_transcripts.py`, [detector_validation.md](data/validation/detector_validation.md) |
| GRPO pipeline runs end-to-end | `tests/smoke_test_grpo*.py` |

### Local smoke JSONs (not V10 — do not cite)

Dev runs with `summary` schema and tiny checkpoints (`baseline_sft_*`, `grpo_tiny_*`, `hh_smoke_*`). See previous naming map in git history; not used for verdicts above.

---

## Limitations

- Single training seed in V9; eval uses 5 inference seeds.
- Transfer trained eval incomplete (15/40).
- H3 persona eval not run at V10 scale.
- Goal-drift detector is embedding-similarity proxy; instruction rules are regex subset.
- Text-only — no audio latency in benchmark.

---

## Artifacts

| Artifact | Location | Status |
|----------|----------|--------|
| V10 eval JSONs | `data/eval_results/in_domain_*.json`, `transfer_*.json`, `reward_hacking_probes.json` | Synced |
| Training log | `data/training_logs/run_v9/episode_log.jsonl` | Synced |
| Figures | `plots/*.png` via `python scripts/make_plots.py` | Generate locally |

---

## Open gaps

1. Complete `transfer_trained.json` to n=40.
2. Run H3 on `eval_held_out_persona.jsonl`.
3. Diagnose train/eval disconnect (ablations: rollout length, detector-off, scripted customer).
4. Multi-seed training for variance estimates.
