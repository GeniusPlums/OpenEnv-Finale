# Role Drift — HF Space (V4)

Gradio UI for the hackathon: **auto-tour** (JSON transcripts, no LLM), **paste prompt** (trained model + optional lazy baseline), **pick scenario** (side-by-side eval JSON).

## Deploy

From a clone of the GitHub repo with `space/` as the app directory (or copy `space/*` into a Space that tracks `OpenEnv-Finale`).

Copy eval artifacts into `space/data/eval_results/` (same filenames as the Hub dataset). Copy `data/scenarios/eval.jsonl` and `data/prompts/*.md` as needed. The app resolves paths relative to this folder; parent-repo layout also works if you run with repo root on `PYTHONPATH`.

**FM-3 / FM-6:** This Space does **not** start vLLM. Customer turns use **scripted personas** only. The app injects a stub for `llm_backed` so `load_llm_persona` is never used on CPU/T4 in production.

**FM-3:** The baseline 1.5B model is loaded **only** after the user clicks the baseline control in Tab 2, not at startup.

## Test locally

```bash
export PYTHONPATH="/path/to/OpenEnv-Finale"
cd space
pip install -r requirements.txt
python app.py
```

## Links

- Model: [GeniusPlums/role-drift-qwen-1-5b-grpo](https://huggingface.co/GeniusPlums/role-drift-qwen-1-5b-grpo)
- GitHub: [github.com/GeniusPlums/OpenEnv-Finale](https://github.com/GeniusPlums/OpenEnv-Finale)
