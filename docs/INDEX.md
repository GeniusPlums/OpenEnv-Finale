# Documentation index

**Role Drift Environment** — OpenEnv-compatible RL for conversational agent reliability.

Use this page to navigate the repository. If you are new, read in the order below.

---

## Start here (5 minutes)

| Step | Document | What you get |
|------|----------|--------------|
| 1 | [README](../README.md) | Problem, environment design, training stack, links to Hub artifacts |
| 2 | [ARCHITECTURE.md](ARCHITECTURE.md) | System diagram, package layout, reward flow, vLLM dependency |
| 3 | [BENCHMARK.md](../BENCHMARK.md) | Pre-registered verdicts (H1/H2/H4 falsified), train vs eval metrics |
| 4 | [REPRODUCTION.md](REPRODUCTION.md) | Step-by-step commands to install, smoke-test, eval, and train |

---

## Research & results

| Document | Audience | Contents |
|----------|----------|----------|
| [RESEARCH_OVERVIEW.md](RESEARCH_OVERVIEW.md) | Researchers, reviewers | Origin story, taxonomy, detector design, training narrative |
| [hypotheses.md](hypotheses.md) | Researchers | Pre-registered H1–H4 claims and falsification criteria |
| [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md) | Eval engineers | Publication roadmap, missing metrics, figure checklist |
| [BENCHMARK.md](../BENCHMARK.md) | Everyone | Headline numbers (filled from JSON only) |
| [data/validation/detector_validation.md](../data/validation/detector_validation.md) | Detector authors | Transcript-level detector evidence |

---

## Engineering & operations

| Document | Audience | Contents |
|----------|----------|----------|
| [REPRODUCTION.md](REPRODUCTION.md) | Anyone reproducing | Install, vLLM, eval harness, HF Jobs, plots |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Maintainers | Honest post-mortem: what broke, what works, blessed paths |
| [REPO_AUDIT.md](REPO_AUDIT.md) | Maintainers | KEEP / ARCHIVE / DELETE classification |
| [../scripts/README.md](../scripts/README.md) | HF Jobs operators | Launcher decision tree |
| [../HH_RUNBOOK_HF.md](../HH_RUNBOOK_HF.md) | HF Jobs (legacy) | A100 mini-smoke gate — see stable H200 path first |

---

## Contributing & community

| Document | Audience | Contents |
|----------|----------|----------|
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Contributors | How to add scenarios, detectors, personas; PR expectations |
| [CONTENT_PLAN.md](CONTENT_PLAN.md) | Writers | Blog/article series outline (not tutorial fluff) |

---

## Historical / reference (not operator docs)

| Document | Notes |
|----------|-------|
| [../CLAUDE.md](../CLAUDE.md) | Original project brief |
| [hackathon_progress.md](hackathon_progress.md) | April snapshot — may lag README V9 claims |
| [project_reference.md](project_reference.md) | Pitch prep — what to claim vs not |
| [pitch_deck.md](pitch_deck.md), [video_script.md](video_script.md), [blog_post.md](blog_post.md) | Demo and outreach drafts |

---

## External artifacts

| Artifact | URL |
|----------|-----|
| Trained model | [GeniusPlums/role-drift-qwen-1-5b-grpo](https://huggingface.co/GeniusPlums/role-drift-qwen-1-5b-grpo) |
| Eval results dataset | [GeniusPlums/role-drift-eval-results](https://huggingface.co/datasets/GeniusPlums/role-drift-eval-results) |
| HF Space (demo) | See [space/README.md](../space/README.md) |
| OpenEnv framework | [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv) |

---

## Quick commands

```bash
# CPU smoke (no GPU)
pip install -e .
pytest tests/ -q -k "not grpo"

# Download publication eval bundle (requires hf CLI + auth)
hf download GeniusPlums/role-drift-eval-results --local-dir data/eval_results

# Regenerate figures (after download)
pip install -e ".[plots]"
python scripts/make_plots.py
```
