# Changelog

All notable changes to this project’s **documented conclusions** and benchmark artifacts.

## [Unreleased] — 2026-06-22

### Changed — project conclusions (benchmark verdict)

Synced V10 eval JSONs and V9 `episode_log.jsonl` from Hugging Face Hub into `data/eval_results/` and `data/training_logs/run_v9/`. Re-ran hypothesis and reward-hacking checks against pre-registered thresholds in `docs/hypotheses.md`.

**Final hypothesis verdicts:**

| ID | Verdict |
|----|---------|
| H1 (in-domain trained > baseline) | **FALSIFIED** |
| H2 (transfer trained > baseline) | **FALSIFIED** |
| H3 (held-out persona) | **SKIPPED** |
| H4 (per-drift improvement order) | **FALSIFIED** |
| Reward-hacking (trained > trivial policies) | **FALSIFIED** |

**Key empirical finding:** Training group-mean return peaked at **3.215** (episode 98) on the training distribution, but held-out in-domain eval mean was **−1.58** vs prompted baseline **+0.19** (n=50 each, non-overlapping 95% CIs). Trivial eval policies outscore the V9 checkpoint.

### Changed — documentation

- **README.md:** Removed implications that GRPO improved held-out performance; split training vs eval metrics; replaced TBD hypothesis table with final verdicts; reframed contribution as benchmark + falsification framework.
- **BENCHMARK.md:** Added final verdict table, training/eval tables, train/eval disconnect section, reward-hacking numbers, synced artifact status.
- **docs/hypotheses.md:** Added verdict summary and per-hypothesis outcomes.
- **docs/RESEARCH_OVERVIEW.md:** Rewrote abstract and §6 results to match V10; updated OpenEnv positioning.
- **docs/REPO_AUDIT.md:** Updated open-source positioning statement.
- **docs/blog_post.md:** Marked draft superseded; replaced false results section with V10 numbers.
- **docs/INDEX.md:** Five-minute path points to verdict table.
- **space/README.md:** Space copy reflects falsification framing, not deployment win.
- **openenv.yaml:** Description updated for benchmark/falsification positioning.

### Not changed

- No new training runs or experiments.
- No changes to detector code, GRPO trainer, or eval harness logic.
- Homepage / Gradio app code not modified (copy drafts only in `space/README.md`).

### Known gaps (unchanged)

- `transfer_trained.json` incomplete (15/40 episodes on Hub).
- H3 persona eval not run at V10 scale.
- Canonical `plots/*.png` not committed (generate via `make_plots.py`).
