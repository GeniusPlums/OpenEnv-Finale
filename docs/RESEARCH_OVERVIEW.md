# Research Overview

A workshop-paper-style summary of the Role Drift Environment research program.

**Status:** Environment, training pipeline, and V10 held-out eval complete. Hub artifacts synced locally (2026-06-22). Pre-registered hypotheses **H1, H2, H4 falsified**; **H3 skipped**.

---

## Abstract

Production conversational agents—especially voice agents constrained to small, low-latency language models—exhibit recurring **behavioral failures** that resist prompt engineering: inability to end calls, abandonment of assigned tasks, violation of explicit instructions, and unprompted language switches. We introduce an **OpenEnv-compatible environment and benchmark** that reproduces these failures in text, scores them with **composable programmatic reward detectors**, and couples them to a **GRPO** training recipe on **Qwen2.5-1.5B** against a frozen **Qwen2.5-7B** adversarial customer simulator. The system is grounded in **real production transcripts** and a scenario corpus with disjoint train, eval, transfer, and held-out persona splits. We pre-register four hypotheses and report results from a 100-episode training run with bootstrap eval on held-out scenarios. **Detector miscalibration causes negative learning in early diagnostics; after calibration, training return improves on the training distribution (peak group-mean 3.215 at episode 98). Held-out evaluation falsifies the primary hypotheses: the trained checkpoint underperforms the prompted baseline in-domain (mean −1.58 vs +0.18) and on partial transfer eval, fails reward-hacking checks against trivial policies, and does not match the pre-registered per-drift improvement ordering.** We release the environment, eval harness, logs, and JSONs as a **falsification framework** for programmatic-reward RL on role drift.

---

## 1. Research question

**Can gradient-based reinforcement learning on composable drift detectors improve role adherence in deployable-class language models beyond prompt-only baselines, under adversarial customer pressure?**

### Sub-questions

1. Do programmatic detectors align with failures in real production transcripts?
2. Does GRPO on episode-level returns produce monotonic improvement without detector gaming?
3. Does improvement on Masters' Union / Kundan Kishore scenarios transfer to a held-out DearConnect prompt domain?
4. Which drift type is easiest/hardest to reduce under shared training?

---

## 2. Motivation and hypothesis

### 2.1 Problem framing

Voice-agent deployments face a **latency ceiling** (~500ms LLM budget). Frontier models exceed this; deployable 1–7B models do not reliably maintain role, instructions, or call-closing behavior over long dialogues. Existing eval products **detect** drift; they do not provide a standard **training environment** with reproducible reward signals.

### 2.2 Core hypothesis (H1)

> A GRPO-trained Qwen2.5-1.5B-Instruct will achieve **higher mean aggregate reward** than the same model with prompt-only instructions on held-out in-domain scenarios, with **non-overlapping 95% bootstrap confidence intervals**.

**Falsification:** Trained mean ≤ baseline within CI overlap.

### 2.3 Transfer hypothesis (H2)

> The trained model **generalizes** to DearConnect transfer scenarios (unseen `prompt_id` at eval time).

**Falsification:** Trained performs equal to or worse than prompted baseline on transfer eval.

### 2.4 Adversarial persona hypothesis (H3)

> The trained model resists a **combined-pressure** held-out persona (off-topic + language switch in one conversation).

**Status:** Pre-registered; **skipped in V10** due to compute constraints. Scenarios exist in `eval_held_out_persona.jsonl`.

### 2.5 Component difficulty hypothesis (H4)

> Per-component reward improvements follow **termination > instruction > goal > language**.

**Falsification:** A lower-ranked component improves more than a higher-ranked one (e.g., language > goal).

**Prior note:** Instruction-heavy scenarios may dominate if explicit rules are easier to learn than semantic goal adherence.

---

## 3. Environment design

### 3.1 Episode structure

- **Agent:** Trainable policy (1.5B class).
- **Customer:** Frozen LLM persona (7B via vLLM) or scripted fallback.
- **System prompt:** Production-derived prompts (`kundan_kishore`, `masters_union`, `dearconnect`).
- **Turn loop:** Agent utterance → reward → customer reply → until `end_call`, farewell resolution, or `max_turns`.

### 3.2 Scenario corpus

| Split | n | Purpose |
|-------|---|---------|
| Train | 40 | GRPO — ~5 scenarios per drift type × 2 domains + cooperative variants |
| Eval (in-domain) | 10 | Held-out IDs, disjoint from train |
| Transfer | 8 | DearConnect domain shift |
| Held-out persona | 4 | H3 (not run in V10) |
| Injection | 4 | Stress tests |

### 3.3 Real-world grounding

Three production failure transcripts inform personas and detector validation:

| Transcript | Domain | Signature failure |
|------------|--------|-------------------|
| Kundan Kishore | Trading workshop sales | 15+ turn thank-you loop; Spanish switch; AI reveal |
| Masters' Union | College admissions | Complete goal drift → startup/real estate consulting |
| DearConnect | Broker platform | Numbered-list dumps; soft-no handling |

Detectors are validated against these transcripts (`data/validation/detector_validation.md`; `tests/test_detectors_on_real_transcripts.py`).

---

## 4. Reward architecture

Total per-turn reward is a **weighted sum** of detector penalties plus anti-gaming bonuses, clipped to [-5, 5].

| Component | Mechanism |
|-----------|-----------|
| Termination drift | Penalize agent turns after customer farewell signals |
| Goal drift | Embedding similarity to `task_description` (threshold 0.18) |
| Instruction drift | Regex rules: deadline mentions, fee phrasing, list format, turn length |
| Language drift | `langdetect` vs conversation baseline |
| Task bonus | Reward clean, concise, penalty-free turns |
| Terminal success | Bonus for successful call closure per outcome predicates |

**Critical calibration event:** Early 20-episode training **degraded** returns. Root cause analysis showed goal threshold too aggressive (87% fire rate) and a language detector bug (constant 0.5 penalty). After fixing (goal 0.35→0.18, language rewrite), a second 20-episode diagnostic **improved** returns. This establishes detector calibration as a first-class research step, not an afterthought.

---

## 5. Experimental setup

### 5.1 Models

| Role | Model | Trainable |
|------|-------|-----------|
| Agent policy | Qwen/Qwen2.5-1.5B-Instruct | Yes |
| Reference (KL) | Copy of initial policy | Frozen |
| Customer sim | Qwen/Qwen2.5-7B-Instruct (vLLM) | No |

### 5.2 Training (V9 configuration)

| Setting | Value |
|---------|-------|
| Algorithm | GRPO (group-relative advantages, episode-as-sample) |
| Episodes | 100 (reported production run) |
| Group size | 4 |
| Max turns per rollout | 6 (memory constraint) |
| Curriculum | Adversarial scenario weighting |
| Infrastructure | Hugging Face Jobs, H200 GPU |

### 5.3 Evaluation (V10 configuration)

| Setting | Value |
|---------|-------|
| Harness | `scripts/run_eval.py` |
| In-domain | 10 scenarios × 5 seeds |
| Transfer | 8 scenarios × 5 seeds |
| Policies | Prompted 1.5B baseline vs GRPO checkpoint |
| Statistics | Bootstrap 95% CIs on episode returns |
| Reward hacking | Four trivial policies compared to trained |

### 5.4 Baselines

| Baseline | Role |
|----------|------|
| **Prompted Qwen2.5-1.5B** | Primary — same model class, no RL |
| SFT (optional) | Demonstration fine-tune — partial implementation |
| Frontier prompted (optional) | Groq/GPT-4o — capability upper bound, not latency-viable |

---

## 6. Results

### 6.1 Verified facts (in-repository evidence)

| Finding | Evidence | Confidence |
|---------|----------|------------|
| Detectors fire on real transcript patterns | `detector_validation.md`, unit tests | High |
| Uncalibrated detectors break learning | Diag1: returns worsened (slope −0.20) | High |
| Calibrated detectors yield learnable **training** signal | Diag2: training return slope +0.14 | High — does not imply held-out eval gain |
| GRPO pipeline runs end-to-end | Smoke tests, `tests/smoke_test_grpo*.py` | High |
| Local tiny eval (4 scenarios, 1 seed) | `grpo_tiny_*` worse than `baseline_sft_*` | High — **not** a headline result (under-trained checkpoint) |

### 6.2 Training metrics (V9 — not eval)

| Metric | Value | Source |
|--------|-------|--------|
| Episodes | 100 | `episode_log.jsonl` |
| Best group-mean training return | **3.215** (episode **98**) | Same |

### 6.3 Held-out evaluation (V10)

| Comparison | Baseline | Trained | Verdict |
|------------|----------|---------|---------|
| In-domain | **0.185** [−0.19, 0.55] | **−1.580** [−1.84, −1.34] | Trained below baseline |
| Transfer | **0.476** [−0.04, 1.04] n=40 | **−0.818** [−0.99, −0.64] n=**15** | Trained below baseline; eval incomplete |

Full tables: [BENCHMARK.md](../BENCHMARK.md).

### 6.4 Train/eval disconnect

Training return rose (diag2 slope +0.14; V9 peak 3.215) while held-out eval **worsened** vs prompted baseline. Trivial eval policies outscore the trained checkpoint. Calibration is **necessary but not sufficient** for held-out improvement under this recipe.

### 6.5 Hypothesis verdict table

| ID | Verdict | Notes |
|----|---------|-------|
| H1 | **FALSIFIED** | Non-overlapping CIs; trained below baseline |
| H2 | **FALSIFIED** | Trained below baseline on partial transfer eval |
| H3 | **SKIPPED** | No V10 persona paired eval |
| H4 | **FALSIFIED** | Per-drift ordering not observed; language Δ > goal on partial transfer |
| Reward-hacking | **FALSIFIED** | Trained −1.58 vs best trivial −0.07 |

### 6.6 Failed / incomplete runs (honest record)

| Run | Outcome | Lesson |
|-----|---------|--------|
| 200-ep L40S job | Canceled ~40 min, no logs | Billing / ops |
| Transfer trained eval | 15/40 episodes | Incomplete artifact on Hub |
| H3 held-out persona | Skipped in V10 | **SKIPPED** hypothesis |

---

## 7. Limitations

### 7.1 Environment fidelity

- **Text-only** — no STT/TTS errors, barge-in, or prosody.
- **Simplified prompts** — production-derived but not full 3500-word deployments in all paths.
- **Customer frozen** — does not model co-adapting users.

### 7.2 Detector fidelity

- **Goal drift** — embedding similarity is a coarse proxy; semantic entailment would be stronger.
- **Instruction drift** — small regex rule set; not full prompt compliance.
- **Language drift** — `langdetect` errors on short utterances; loanword handling is heuristic.

### 7.3 Experimental rigor

- **Single training seed** in V9 — inference is multi-seed, training is not.
- **Short training rollouts** (6 turns) vs longer eval (up to 30) — distribution shift risk.
- **Persona fallback** — if vLLM fails, scripted customer changes difficulty.

### 7.4 Operational

- **GPU co-location** — vLLM 7B + 1.5B training limits reproducibility on smaller hardware.
- **Detector versioning** — no automated freeze between train and eval beyond discipline.

---

## 8. Future directions

### Near-term (reproducibility)

1. Diagnose train/eval disconnect; complete transfer eval (15→40).
2. Consolidate HF Job launchers to one blessed path.
3. Log per-detector episode means in training for learning-curve decomposition.

### Medium-term (science)

1. Run H3 held-out persona eval.
2. Multi-seed training (≥3) for stability claims.
3. Ablations: detector-off, scripted-vs-LLM customer, SFT-vs-GRPO.
4. Stronger goal detector (LLM judge or NLI) on ambiguous cases.

### Long-term (impact)

1. Scale recipe to Maverick-class 17B — same environment, larger policy.
2. Community scenario contributions — new domains, drift types, locales.
3. Integration with TRL `GRPOTrainer` and OpenEnv hub listing as reference environment.
4. Latency-aware evaluation — tokens/sec and wall-clock per turn alongside reward.

---

## 9. Related work (positioning)

| Area | This project |
|------|--------------|
| Conversational eval SDKs (Coval, Hamming, Cekura) | Detection only → **benchmark + falsification harness** |
| General RLHF / GRPO | Domain-specific **drift compositors**; V9 run shows train/eval gap |
| OpenEnv environments | Reference **agent reliability** env with pre-registered negative result |
| Prompt robustness benchmarks | **Behavioral** attractors; held-out eval can falsify training claims |

---

## 10. Citation (proposed)

```bibtex
@software{role_drift_env2026,
  title        = {Role Drift Environment: OpenEnv RL for Conversational Agent Reliability},
  author       = {GeniusPlums},
  year         = {2026},
  url          = {https://github.com/GeniusPlums/OpenEnv-Finale},
  note         = {OpenEnv-compatible environment for training against role drift}
}
```

---

## Appendix: Pre-registration

Full hypothesis statements: [hypotheses.md](hypotheses.md)  
Benchmark fill-in procedure: [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md)  
Architecture detail: [ARCHITECTURE.md](ARCHITECTURE.md)
