# Role Drift — HF Space

Gradio demo for the **Role Drift benchmark**: production failure transcripts, scenario picker, and **side-by-side V10 eval JSON** (baseline vs trained on held-out scenarios).

**Framing:** This Space demonstrates the **environment and falsification results**, not a deployment-ready trained agent. V10 held-out eval **falsified** H1/H2 — the trained checkpoint scores below the prompted baseline on synced Hub JSONs. See [BENCHMARK.md](../BENCHMARK.md).

## Tabs

1. **Auto-tour** — JSON transcripts of real production failures (no LLM).
2. **Paste prompt** — optional lazy-loaded baseline vs V9 checkpoint (educational; not eval-grade on Space hardware).
3. **Pick scenario** — compare `in_domain_baseline.json` vs `in_domain_trained.json` aggregates from Hub.

## Deploy

From a clone of the GitHub repo with `space/` as the app directory (or copy `space/*` into a Space that tracks `OpenEnv-Finale`).

Copy eval artifacts into `space/data/eval_results/` (same filenames as the Hub dataset `GeniusPlums/role-drift-eval-results`). Copy `data/scenarios/eval.jsonl` and `data/prompts/*.md` as needed.

**Constraints:** This Space does **not** start vLLM. Customer turns use **scripted personas** only. The baseline 1.5B model loads only after user action, not at startup.

## Test locally

```bash
export PYTHONPATH="/path/to/OpenEnv-Finale"
cd space
pip install -r requirements.txt
python app.py
```

## Links

- Benchmark: [BENCHMARK.md](../BENCHMARK.md)
- Model artifact (V9): [GeniusPlums/role-drift-qwen-1-5b-grpo](https://huggingface.co/GeniusPlums/role-drift-qwen-1-5b-grpo)
- Eval JSONs: [GeniusPlums/role-drift-eval-results](https://huggingface.co/datasets/GeniusPlums/role-drift-eval-results)
- GitHub: [github.com/GeniusPlums/OpenEnv-Finale](https://github.com/GeniusPlums/OpenEnv-Finale)
- OpenEnv: [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)
