# Role Drift Eval - Benchmark

A benchmark for measuring voice-agent resistance to behavioral drift under adversarial conversational pressure.

## Quick Start

```bash
pip install -e .
role-drift-eval --model-path Qwen/Qwen2.5-1.5B-Instruct --scenarios data/scenarios/eval.jsonl
```

## Benchmark Details

- 10 in-domain eval scenarios (from training domain)
- 8 transfer scenarios (DearConnect domain - different prompt)
- 4 held-out adversarial persona scenarios (combined pressure)
- 4 prompt injection eval scenarios

Total: 26 scenarios. Run time: ~10 minutes on single GPU.

## Leaderboard

| Model | In-Domain | Transfer | Held-Out Persona | Notes |
|-------|-----------|-----------|------------------|-------|
| Prompted Qwen1.5B | -2.1 | pending | pending | Baseline |
| (Trained checkpoint) | | | | Pending compute |

## Validity Caveats

- Customer simulator quality affects eval realism
- Eval scale is small (n=10-26 per domain)
- Prompt domains only cover 3: Kundan Kishore, Masters' Union, DearConnect

## Contact

For benchmark additions or new domain prompts, open an issue.