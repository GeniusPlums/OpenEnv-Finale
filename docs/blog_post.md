# Role Drift: Benchmarking Conversational Agent Drift

> **Status (2026-06-22):** Draft superseded by synced V10 eval. Pre-registered hypotheses **H1, H2, H4 falsified**; see [BENCHMARK.md](../BENCHMARK.md). Do not use the "Results" section below — it reflects an aspirational narrative, not held-out eval.

Production voice agents are stuck between two bad choices: frontier LLMs that are too slow for real-time voice, and small models that drift out of role the moment a conversation gets interesting.

We built an OpenEnv-compatible **benchmark and environment** that measures this drift with composable detectors and pre-registered held-out evaluation.

## The Four Drifts

After analyzing hundreds of production call transcripts, four failure modes show up again and again:

1. **Termination drift** — The customer says "thanks, bye" fifteen times. The agent says "you're welcome" fifteen times. Neither hangs up.
2. **Goal drift** — The agent was supposed to sell a trading workshop. It ends up giving real estate advice.
3. **Instruction drift** — The system prompt says "one idea per turn, no numbered lists." The agent replies with "1. ... 2. ... 3. ..."
4. **Language drift** — Mid-conversation, the customer switches to Spanish. The agent follows.

## Why Prompts Don't Fix It

We have a 3500-word production prompt with a "Final Rule (Read This Twice)" section. It fails on all four. Behavioral attractors don't respond to more words.

## The Environment

Our environment simulates adversarial customers designed to trigger each drift type. Four composable reward detectors score every agent turn:

- **Termination:** Detects farewell signals + disengagement counter
- **Goal:** Embedding similarity to task description
- **Instruction:** Regex/rule-based checkers for prompt-specific rules
- **Language:** Language ID with loanword whitelist

Plus a terminal success bonus for episode-level outcome predicates.

## Results (V10 — held-out eval)

After syncing Hub eval JSONs (2026-06-22):

- **Training (V9):** 100 episodes; best group-mean training return **3.215** at episode 98.
- **Held-out in-domain eval:** Prompted baseline **+0.19** vs GRPO checkpoint **−1.58** (n=50 each). **H1 falsified.**
- **Transfer eval:** Baseline **+0.48** vs trained **−0.82** (trained n=15/40 incomplete). **H2 falsified** on available data.
- **Reward-hacking:** Trivial policies outscore the trained checkpoint on eval. **Falsified.**

**Lesson:** Detector calibration enables learnable training signal (diag2), but this GRPO recipe did **not** improve held-out role adherence. The value is a **falsification framework**, not a deployment win.

## What's Next

- Diagnose train/eval disconnect (rollout length, reward gaming, distribution shift)
- Complete transfer eval; run H3 persona eval
- Community scenarios and detector improvements

---

*Built for OpenEnv Hackathon India 2026. Environment, detectors, and training code at [GitHub link]. Hosted on Hugging Face Spaces.*
