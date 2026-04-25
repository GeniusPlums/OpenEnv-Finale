# Pre-Registered Hypotheses

These hypotheses were registered before any training runs. Results will be reported honestly.

## H1: Trained vs Prompted In-Domain
**Claim**: A GRPO-trained Qwen2.5-1.5B will achieve higher mean aggregate reward than the same model with prompt-only instructions.
- **Threshold**: Trained model mean > prompted baseline, with non-overlapping 95% CIs
- **Falsification**: Trained ≤ prompted within CI overlap

## H2: Transfer to DearConnect Domain
**Claim**: Trained model generalizes to DearConnect transfer scenarios (prompt_id: dearconnect).
- **Threshold**: Trained model on transfer eval beats prompted-baseline on transfer eval
- **Fabsification**: Trained performs equal or worse than prompted on transfer domain

## H3: Held-Out Adversarial Persona
**Claim**: Trained model resists combined_pressure persona (off-topic + language switch in same conversation).
- **Threshold**: Trained > prompted on eval_held_out_persona scenarios
- **Falsification**: No improvement on held-out persona

## H4: Component Difficulty Order
**Claim**: Per-component reward improvements follow termination > instruction > goal > language.
- **Threshold**: Termination drift most improved (easiest to learn), language least
- **Falsification**: Language improves more than goal

---

## How to Interpret Results

| Outcome | Meaning |
|---------|---------|
| HELD | Threshold met, hypothesis supported |
| FALSIFIED | Threshold not met, hypothesis rejected |
| INCONCLUSIVE | Insufficient compute/data to determine |

All results will be reported honestly regardless of outcome.