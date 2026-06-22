# Pre-Registered Hypotheses

These hypotheses were registered before training runs. Verdicts below are from synced V10 eval JSONs (`data/eval_results/`, 2026-06-22). See [BENCHMARK.md](../BENCHMARK.md) for full numbers.

## Verdict summary

| ID | Verdict |
|----|---------|
| H1 | **FALSIFIED** |
| H2 | **FALSIFIED** |
| H3 | **SKIPPED** |
| H4 | **FALSIFIED** |

---

## H1: Trained vs Prompted In-Domain

**Claim**: A GRPO-trained Qwen2.5-1.5B will achieve higher mean aggregate reward than the same model with prompt-only instructions.

- **Threshold**: Trained model mean > prompted baseline, with non-overlapping 95% CIs
- **Falsification**: Trained ≤ prompted within CI overlap
- **Verdict**: **FALSIFIED** — trained **−1.580** [−1.84, −1.34] vs baseline **0.185** [−0.19, 0.55]; n=50; CIs do not overlap; trained below baseline

## H2: Transfer to DearConnect Domain

**Claim**: Trained model generalizes to DearConnect transfer scenarios (prompt_id: dearconnect).

- **Threshold**: Trained model on transfer eval beats prompted-baseline on transfer eval
- **Falsification**: Trained performs equal or worse than prompted on transfer domain
- **Verdict**: **FALSIFIED** — trained **−0.818** vs baseline **0.476** on available data; trained eval incomplete (15/40 episodes)

## H3: Held-Out Adversarial Persona

**Claim**: Trained model resists combined_pressure persona (off-topic + language switch in same conversation).

- **Threshold**: Trained > prompted on eval_held_out_persona scenarios
- **Falsification**: No improvement on held-out persona
- **Verdict**: **SKIPPED** — no V10 paired baseline/trained eval on `eval_held_out_persona.jsonl`

## H4: Component Difficulty Order

**Claim**: Per-component reward improvements follow termination > instruction > goal > language.

- **Threshold**: Termination drift most improved (easiest to learn), language least
- **Falsification**: Language improves more than goal
- **Verdict**: **FALSIFIED** — in-domain Δ (trained−baseline): term −3.60, instr −2.84, goal −1.62; transfer partial Δ: language **+1.51** > goal −2.08

---

## Reward-hacking check (not pre-registered as H5)

Trained in-domain eval mean (**−1.58**) is below all four trivial policies on `eval.jsonl` (best trivial **−0.07**). **FALSIFIED** as an anti-gaming sanity check.

---

## How to Interpret Results

| Outcome | Meaning |
|---------|---------|
| HELD | Threshold met, hypothesis supported |
| FALSIFIED | Threshold not met, hypothesis rejected |
| INCONCLUSIVE | Insufficient compute/data to determine |
| SKIPPED | Pre-registered but not run at benchmark scale |

All results reported regardless of outcome.
