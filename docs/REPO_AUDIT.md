# Repository Audit

**Date:** 2026-06-22  
**Scope:** Full tree under `OpenEnv-Finale` — code, scripts, data, docs, artifacts  
**Purpose:** Transition from hackathon submission to maintainable open-source research engineering project

---

## Executive summary

The **core research artifact is sound**: an OpenEnv-compatible environment, four composable drift detectors, scenario corpus, GRPO training loop, and V10 eval harness. The main liabilities are **operational sprawl** (8+ HF Job launchers), **documentation drift** (README ahead of local `BENCHMARK.md` and `plots/`), **root-level clutter** (debug scripts and one-off tests), and **uncommitted infra** (`run_stable_h200.sh`, CUDA bootstrap split) that remote jobs cannot use until pushed.

**Recommended next engineering action (before more docs):** commit and push a single blessed HF entry path (`scripts/run_stable_h200.sh` + `launch_hf_stable_h200_job.py`), archive superseded launchers, and download Hub eval JSONs into `data/eval_results/` so `make_plots.py` can run locally.

---

## Inventory at a glance

| Area | Count / state | Health |
|------|----------------|--------|
| Environment package (`role_drift_env/`) | ~15 modules | **Keep** — production-quality |
| Training (`training/`) | 10 modules | **Keep** — refactor entry split |
| Tests (`tests/`) | 8 files | **Keep** — expand eval regression |
| Scripts (`scripts/`) | 61 files | **Refactor** — heavy duplication |
| Scenarios | 66 total (40 train / 10 eval / 8 transfer / 4 held-out / 4 injection) | **Keep** |
| Eval JSONs (`data/eval_results/`) | 11 files — mostly smoke/tiny, not V10 Bundle A | **Refactor** — align naming with BENCHMARK.md |
| Training logs (`data/training_logs/`) | Smoke checkpoints only; no `run_v9/episode_log.jsonl` in repo | **Gap** — pull from Hub |
| Plots (`plots/`) | Only `diag2/` (2 PNGs); README references 4 missing figures | **Gap** |
| Root-level `test_*.py` | 4 files outside `tests/` | **Delete or move** |
| Docs | 25 markdown files; overlapping hackathon + ops narratives | **Refactor** |

---

## Keep

### Core package

| Path | Role |
|------|------|
| `role_drift_env/models.py` | Action, Observation, State, Scenario dataclasses |
| `role_drift_env/client.py` | OpenEnv HTTP client |
| `role_drift_env/server/environment.py` | `reset` / `step` / terminal success |
| `role_drift_env/server/customer_sim.py` | Persona router |
| `role_drift_env/server/personas/` | LLM-backed + scripted personas |
| `role_drift_env/server/rewards/` | Four detectors + composer + terminal success |
| `role_drift_env/server/app.py` | FastAPI server for HF Spaces |

### Training & evaluation

| Path | Role |
|------|------|
| `training/train_grpo_core.py` | GRPO trainer (episode-as-sample, group advantages) |
| `training/train_grpo.py` | CUDA-safe entry (after bootstrap push) |
| `training/rollout.py` | Shared episode loop for train + eval |
| `training/eval_baseline.py` | Baseline policies (local model, Groq API) |
| `scripts/run_eval.py` | V10 eval harness with bootstrap CIs |
| `scripts/make_plots.py` | Publication figure generator |
| `scripts/wait_for_cuda.py` | H200 NVSwitch fabric wait |
| `scripts/compare_eval_runs.py` | Diff eval JSON summaries |

### Data & validation

| Path | Role |
|------|------|
| `data/scenarios/*.jsonl` | Train / eval / transfer splits |
| `data/prompts/*_full.md` | Production-style system prompts |
| `data/prompts/rules/*.json` | Instruction-drift rule specs |
| `data/transcripts/*.txt` | Real failure transcripts |
| `data/validation/detector_validation.md` | Detector vs transcript evidence |
| `data/personas/adversarial_customers.json` | Persona definitions |

### Tests

| Path | Role |
|------|------|
| `tests/test_detectors_on_real_transcripts.py` | Regression on production failures |
| `tests/test_language_detector.py` | Language detector unit tests |
| `tests/test_rollout_loop.py` | Environment integration |
| `tests/smoke_test_grpo*.py` | GRPO pipeline smokes |
| `tests/conftest.py` | Shared fixtures |

### Documentation (retain, consolidate)

| Path | Role |
|------|------|
| `README.md` | Public entry (rewrite in progress) |
| `BENCHMARK.md` | Numbered results + methodology |
| `docs/ARCHITECTURE.md` | System design |
| `docs/RESEARCH_OVERVIEW.md` | Workshop-paper narrative |
| `docs/hypotheses.md` | Pre-registered hypotheses |
| `docs/PROJECT_STATUS.md` | Honest ops post-mortem |
| `CLAUDE.md` | Original project brief (historical) |

### Single blessed remote path (after push)

| Path | Role |
|------|------|
| `scripts/run_stable_h200.sh` | One in-container entry for training |
| `scripts/launch_hf_stable_h200_job.py` | One local launcher |
| `bootstrap_cuda.py` | First-Python CUDA gate |
| `scripts/cuda_guard.py` | Import-time CUDA guard |

---

## Archive

Move to `archive/` or `docs/archive/` — preserve git history context, remove from default developer path.

### Superseded HF launchers

| File | Superseded by | Notes |
|------|---------------|-------|
| `scripts/launch_v9_job.py` | `launch_hf_v9_training_job.py` | Duplicate V9 launcher |
| `scripts/launch_hf_training_smoke.py` | `launch_hf_smoke_job.py` | Overlapping 2-ep smoke |
| `scripts/launch_hf_full_training.py` | `run_stable_h200.sh` | Inline bash duplicates V9/stable |
| `scripts/launch_hfjob.sh` | `launch_hf_*_job.py` | Shell wrapper without auth helpers |
| `scripts/closeout_v8_runtime.sh` | V9/V10 scripts | V8-era closeout |
| `scripts/hf_job_smoke_inline.sh` | `hf_mini_smoke.sh` | Inline duplicate |

### Hackathon / pitch artifacts (reference only)

| File | Notes |
|------|-------|
| `docs/pitch_deck.md` | Pitch script — not operator docs |
| `docs/video_script.md` | Demo script |
| `docs/blog_post.md` | Draft blog — supersede with CONTENT_PLAN |
| `docs/hackathon_progress.md` | April snapshot; stale vs README V9 claims |
| `IMPLEMENTATION_PLAN.md` | Early roadmap — largely complete |
| `DEV_BRIEF.md` | Duplicate of CLAUDE.md thesis |
| `HH_RUNBOOK.md` | Local A100 runbook — predates H200 stable path |
| `scripts/200ep_post_mortem.md` | Billing post-mortem — move to `docs/ops/` |
| `scripts/session_inventory.md` | Session snapshot — move to `docs/ops/` |
| `scripts/200ep_jobid.txt` | Ephemeral job ID |
| `scripts/detector_diagnostic_report.md` | Valuable analysis — move to `docs/reports/` |

### Prompt archives (already isolated)

| Path | Notes |
|------|-------|
| `data/prompts/_archive/` | 6-line simplified prompts — keep for history |

### Experimental smoke checkpoints

| Path | Notes |
|------|-------|
| `data/training_logs/hh_smoke_train/` | Tokenizer dumps from HF smokes — not a training run |
| `data/training_logs/hh_smoke_train_v3/` | Same |
| `data/training_logs/hh_smoke_train_v4/` | Same |

Do not delete without confirming Hub has canonical `episode_log.jsonl`.

---

## Delete

Safe to remove after confirming no imports reference them.

### Root-level one-off tests (belong in `tests/` or are obsolete)

| File | Reason |
|------|--------|
| `test_mu_instruction.py` | Ad-hoc; coverage in `tests/test_detectors_on_real_transcripts.py` |
| `test_instruction_rules.py` | Ad-hoc regex probe |
| `test_prompt_load.py` | Ad-hoc prompt loader check |
| `test_json.py` | Trivial JSON sanity |

### Debug / scratch artifacts

| File | Reason |
|------|--------|
| `output_diag.txt` | Scratch output |
| `scripts/200ep_jobid.txt` | Ephemeral |
| `scripts/debug_regex.py` | One-off during detector fix |
| `scripts/debug_instr.py` | One-off |
| `scripts/debug_episode.py` | Useful for dev — **optional keep** behind `scripts/dev/` |
| `scripts/cpu_speed_test.py` | Feasibility probe — archive |
| `scripts/kaggle_smoke_test.py` | Wrong platform; HF Jobs is canonical |
| `scripts/check_zip.py` | HF zip debug — archive |
| `scripts/check_error.py` | HF zip debug — archive |
| `scripts/analyze_diag.py` | Diag1 analysis — archive after report preserved |
| `scripts/analyze_diag2.py` | Same |
| `scripts/plot_diag2.py` | Superseded by `make_plots.py` |
| `scripts/plot_training.py` | Superseded by `plot_grpo_training_log.py` / `make_plots.py` |
| `scripts/plot_results.py` | Superseded |
| `scripts/regenerate_baseline_vs_trained_plot.py` | One-off regen — fold into `make_plots.py` |
| `scripts/regenerate_in_domain_vs_transfer_plot.py` | Same |

### Duplicate plot scripts (consolidate to one)

Keep: `scripts/make_plots.py`, `scripts/plot_grpo_training_log.py`  
Delete after merge: `plot_training.py`, `plot_results.py`, `plot_diag2.py`, `regenerate_*.py`

---

## Refactor

### 1. Launcher consolidation (critical)

**Problem:** Eight ways to start HF Jobs; easy to submit V9 while debugging stable path.

| Tier | Script | Use case |
|------|--------|----------|
| **Production training** | `launch_hf_stable_h200_job.py` → `run_stable_h200.sh` | 100+ ep GRPO on H200 |
| **Production eval** | `run_v10_eval_job.sh` → `run_v10_eval_worker.sh` | Bundle A eval |
| **CI / smoke** | `launch_hf_smoke_job.py` or `hf_mini_smoke.sh` | 2-ep pipeline check |
| **Deprecated** | Everything else | Archive with README warning |

Add `scripts/README.md` with a decision tree.

### 2. Eval results naming

`BENCHMARK.md` expects:

- `in_domain_baseline.json`, `in_domain_trained.json`
- `transfer_baseline.json`, `transfer_trained.json`
- `reward_hacking_probes.json`

Local files use different names (`baseline_sft_*`, `grpo_tiny_*`, `hh_smoke_*`). **Refactor:** symlink or rename after Hub download; document mapping in `BENCHMARK.md`.

### 3. Training entry split

| Current | Target |
|---------|--------|
| `train_grpo.py` + `train_grpo_core.py` + `bootstrap_cuda.py` | Single documented entry; CPU path via `ROLE_DRIFT_ALLOW_GRPO_CPU=1` |
| `train_grpo_hfjobs.py` | Merge upload logic into `hub_upload.py` only |

### 4. Documentation map

| Current sprawl | Target |
|----------------|--------|
| `CLAUDE.md`, `DEV_BRIEF.md`, `IMPLEMENTATION_PLAN.md` | `docs/ORIGIN.md` (story) + link from README |
| `HH_RUNBOOK.md`, `HH_RUNBOOK_HF.md` | `docs/ops/HF_JOBS.md` |
| `docs/project_reference.md` | Merge honest claims into `RESEARCH_OVERVIEW.md` |

### 5. `pyproject.toml`

- Add optional extras: `[plots]`, `[vllm]`, `[dev]`
- Pin minimum `trl`, `openenv-core` versions used in production runs
- Add `[project.urls]` for Hub model, dataset, Space

### 6. `.gitignore` vs reproducibility

`plots/*.png` is gitignored — figures cannot ship in-repo without `!plots/README.md` exception or LFS. **Refactor:** commit small canonical figures or document Hub-only artifact policy explicitly.

### 7. Per-component reward logging

`TurnReward.components` exists but `episode_log.jsonl` logs aggregate `mean_return` only. **Refactor:** log `per_detector_mean` per episode for learning-curve breakdown plots.

---

## Dead code & unused files

| Item | Evidence | Action |
|------|----------|--------|
| `training/rollout_diagnostic.py` | Standalone diag script; not imported | Archive |
| `training/generate_sft_data.py` | SFT path partially implemented; eval uses `train_sft` backend label | Keep if SFT baseline planned |
| `training/train_sft.py` | Referenced by `baseline_sft_*` eval JSONs | Keep |
| `scripts/create_tiny_model.py` | Smoke model creation | Move to `scripts/dev/` |
| `scripts/create_hand_labels.py` | Incomplete labeling tool | Archive or finish |
| `scripts/generate_rollout_data.py` | One-off for fire-rate analysis | Archive |
| `scripts/print_closeout_headlines.py` | Pitch helper | Archive |
| `space/` | HF Space stub; README says "coming soon" | Keep; finish Bundle C |
| `bootstrap_cuda.py` (untracked) | Required by local `train_grpo.py` | **Push** |

---

## Documentation inconsistencies

| Claim | Source A | Source B | Resolution |
|-------|----------|----------|------------|
| Best training return ~3.215 | README, BENCHMARK.md | `hackathon_progress.md` says "no final checkpoint" | Trust Hub `episode_log.jsonl` on model repo; cite episode index |
| V10 eval complete | README | `BENCHMARK.md` all TBD; local JSONs are smokes | Download Hub dataset; fill BENCHMARK |
| 200-ep run | `200ep_post_mortem.md` | Canceled, no artifacts | Document as failed scale-up attempt |
| Detector weights in README | `project_reference.md` says -0.5/-0.3 | `composer.py` uses 0.6/1.0/1.0/0.5 weights | **Fix docs** — cite `composer.py` as source of truth |
| Production prompts "3500 words" | CLAUDE.md | `*_full.md` are long but simplified vs raw transcripts | Honest: "production-derived prompts" |
| `run_stable_h200.sh` exists on remote | `launch_hf_stable_h200_job.py` | `PROJECT_STATUS.md`: not on `origin/master` | Push or README warns |

---

## Missing benchmarks & evaluation outputs

### Present locally (verified)

| File | What it is | Publication-ready? |
|------|------------|-------------------|
| `baseline_sft_indomain.json` | 4 scenarios, 1 seed, SFT backend | **No** — tiny smoke |
| `grpo_tiny_indomain.json` | Same 4 scenarios, tiny GRPO checkpoint | **No** — shows regression vs SFT on subset |
| `hh_smoke_eval_*.json` | 1–2 scenario smokes | **No** |
| `reward_hacking_probe.json` | 3 trivial policies, `trained: null` | **Incomplete** |
| `plots/diag2/reward_curve_aggregate.png` | 20-ep diag2 only | **Internal** — not V9 |

### Expected for V10 (per BENCHMARK.md) — missing locally

| Artifact | Status |
|----------|--------|
| `in_domain_baseline.json` | Not in repo (Hub: `GeniusPlums/role-drift-eval-results`) |
| `in_domain_trained.json` | Not in repo |
| `transfer_baseline.json` | Not in repo |
| `transfer_trained.json` | Not in repo |
| `reward_hacking_probes.json` | Partial (`reward_hacking_probe.json` — different schema) |
| `data/training_logs/run_v9/episode_log.jsonl` | Not in repo |
| `plots/reward_curve.png` | Referenced in README, missing |
| `plots/eval_comparison.png` | Missing |
| `plots/transfer.png` | Missing |
| `plots/per_drift_type_curve.png` | Missing |

### Missing metrics (for publication)

- Per-drift-type bootstrap CIs (harness supports `by_drift_type`; not in README tables)
- Multi-seed training variance (only single V9 seed)
- Reward-hacking probe comparison vs trained aggregate
- H3 held-out persona eval (skipped in V10)
- Latency / tokens-per-turn distributions
- Persona fallback rate (`IS_FALLBACK`) during eval
- KL divergence and loss curves from full 100-ep log

See **[BENCHMARK_PLAN.md](BENCHMARK_PLAN.md)** for the publication-quality roadmap.

---

## Open-source positioning (recommended)

### Primary: **OpenEnv benchmark and falsification framework for agent reliability**

**Justification:**

1. **Differentiation:** Eval products detect drift; this repo provides a **reproducible benchmark** that can **falsify** training claims — demonstrated by V10 (H1/H2/H4 falsified on synced Hub JSONs).
2. **Framework fit:** OpenEnv Gymnasium-style API + composable detectors + pre-registered hypotheses.
3. **Honest negative result:** Training return rose (V9 peak 3.215) while held-out eval worsened — a publishable train/eval disconnect.
4. **Avoid overclaiming:** Not a deployment-ready trained agent; the V9 checkpoint is an **artifact** for reproduction.

### Positioning statement (one line)

> **Role Drift Environment** — an OpenEnv benchmark for conversational agent drift with composable reward detectors, pre-registered held-out eval, and a documented GRPO recipe whose V9 run falsified primary training hypotheses.

---

## Priority action list

| Priority | Action | Owner |
|----------|--------|-------|
| P0 | Push `run_stable_h200.sh`, bootstrap, cuda_guard, train_grpo split | Engineering |
| P0 | Download Hub eval JSONs; run `make_plots.py` | Engineering |
| P1 | Archive duplicate launchers; add `scripts/README.md` | Engineering |
| P1 | Fill `BENCHMARK.md` from JSONs (no invented numbers) | Research |
| P1 | Move root `test_*.py` into `tests/` or delete | Engineering |
| P2 | Log per-detector means in `episode_log.jsonl` | Engineering |
| P2 | Commit canonical plots or document Hub-only policy | Research comms |
| P3 | Deploy Gradio Space (Bundle C) | Product |

---

## Audit methodology

This audit used: full file tree listing, README/BENCHMARK/PROJECT_STATUS cross-read, eval JSON inspection, scenario counts, launcher script comparison, git status snapshot (uncommitted CUDA/stable path), and `docs/project_reference.md` implementation checklist. No benchmark numbers were invented; gaps are listed explicitly.
