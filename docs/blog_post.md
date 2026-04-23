# Role Drift: Teaching Small Models to Stay in Character

Production voice agents are stuck between two bad choices: frontier LLMs that are too slow for real-time voice, and small models that drift out of role the moment a conversation gets interesting.

We built an OpenEnv-compatible training environment that turns this drift into a gradient.

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

## Training Recipe

1. **SFT warm-start:** Run a frontier model against the customer sim, keep the top 30% of conversations by reward, fine-tune Qwen2.5-1.5B-Instruct for 1 epoch.
2. **GRPO:** Group-relative policy optimization (G=4) starting from the SFT checkpoint, 200 episodes.
3. **Eval:** Compare against frontier-prompted and deployable-prompted baselines on held-out scenarios.

## Results

The trained 1.5B model beats the deployable-class baseline on aggregate episode return, and approaches the frontier baseline on termination and instruction drift.

## What's Next

- Scale to Llama 4 Maverick-class models (17B) with the same recipe
- LLM-backed customer personas for richer, less scripted adversarial behavior
- Live deployment on Vapi / Bolna stacks

---

*Built for OpenEnv Hackathon India 2026. Environment, detectors, and training code at [GitHub link]. Hosted on Hugging Face Spaces.*
