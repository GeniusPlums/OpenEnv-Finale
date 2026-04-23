# Role Drift Environment

OpenEnv-compatible RL environment that reproduces **role drift** in production voice agents on demand, with programmatic reward signals, so a small open model can be trained to resist drift better than a prompted frontier model.

---

## Problem

Production voice agents built on top of frontier LLMs drift out of their assigned role under conversational pressure. Four recurring failure modes:

1. **Termination drift** — get stuck in thank-you loops, can't end calls
2. **Goal drift** — abandon their assigned task and become a generic helpful assistant
3. **Instruction drift** — violate explicit, simple system-prompt rules
4. **Language drift** — switch languages unprompted

These are **behavioral patterns, not knowledge gaps**. Prompts cannot fix them reliably.

## Origin Story

This project came out of a real production deployment. A client — a real estate broker with **500 cold-callers** — asked for voice-agent automation. The hard part was the LLM: frontier models add latency that breaks real-time voice, fast small models drift. This environment is the training-side answer.

## Environment

- **Episode** = one full agent ↔ customer conversation
- **Action** = agent's next utterance (+ optional `end_call` signal)
- **Observation** = customer reply + turn index + system prompt
- **Reward** = weighted sum of four drift detectors + task bonus + terminal success
- **Done** = agent ends call, max turns reached, or customer fully disengaged

### Detectors

| Detector | Signal | Implementation |
|---|---|---|
| `termination_drift` | Agent talks past clear customer goodbye | Farewell keyword + disengagement counter |
| `goal_drift` | Agent answers off-topic | Embedding similarity to task description |
| `instruction_drift` | Violates explicit prompt rules | Regex/rule-based checkers |
| `language_drift` | Switches language unprompted | fasttext/langdetect with loanword whitelist |
| `terminal_success` | Episode-level outcome predicates | Regex/phrase-match on full transcript |

## Repository Structure

```
role-drift-env/
├── data/
│   ├── prompts/          # Production system prompts
│   ├── transcripts/      # Redacted failure transcripts
│   ├── personas/         # Adversarial customer definitions
│   ├── scenarios/        # 40 train + 10 eval scenarios
│   ├── sft/              # SFT warm-start conversations
│   └── validation/       # Hand-labeled detector ground truth
├── role_drift_env/
│   ├── models.py
│   ├── client.py
│   └── server/
│       ├── environment.py
│       ├── customer_sim.py
│       ├── personas/
│       ├── rewards/
│       └── app.py
├── training/
│   ├── generate_sft_data.py
│   ├── train_sft.py
│   ├── train_grpo.py
│   ├── rollout.py
│   └── eval.py
├── tests/
│   ├── smoke_test_termination.py
│   ├── test_detectors_on_real_transcripts.py
│   └── test_eval_frozen.py
└── docs/
    └── blog_post.md
```

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -e .

# Run smoke test
pytest tests/smoke_test_termination.py -v

# Generate scenarios
python scripts/generate_scenarios.py

# Generate SFT warm-start data
python training/generate_sft_data.py

# Run evaluation
python training/eval.py --model-path checkpoints/sft --scenario-file data/scenarios/eval.jsonl
```

## Training

### SFT Warm-Start

```bash
python training/generate_sft_data.py  # ~400 conversations, top 30% kept
python training/train_sft.py          # 1 epoch on Qwen2.5-1.5B-Instruct
```

### GRPO

```bash
python training/train_grpo.py --episodes 200 --group-size 4
```

## Evaluation

Eval protocol: 10 eval scenarios × 3 seeds × 4 models:

- **Frontier prompted** (GPT-4o / Claude Sonnet)
- **Deployable prompted** (Llama 3.1 8B Instruct)
- **SFT-only** (Qwen 1.5B after warm-start)
- **GRPO-trained** (Qwen 1.5B after GRPO)

Run:
```bash
python training/eval.py --model-path checkpoints/grpo/best --num-seeds 3
```

## Results

See `plots/` for reward curves and eval comparisons.

## Deployment

- **HF Space:** [Link TBD]
- **Video:** [Link TBD]
- **Blog post:** `docs/blog_post.md`

## Citation / License

MIT License. Built for the OpenEnv Hackathon India 2026.

## Known Gaps & Limitations

- Customer simulator uses scripted personas; LLM-backed personas require local vLLM/TGI
- Fasttext language detection not available on Windows Python 3.14; falls back to langdetect
- GRPO training loop is simplified (episode-level rather than full token-level GRPO)
- Evaluation against frontier models requires API keys (not included)
- Only 50 scenarios; production would need 500+
