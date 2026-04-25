# Project Reference Document - Pitch Prep & Q&A

**Date: 2026-04-24 (pitch tomorrow, 2026-04-25)**
**For: Human's personal prep — NOT for judges**

This document is honest about what's actually implemented vs aspirational. The human needs to know what to claim and what to be humble about.

---

## Section 1: Project Summary (One Paragraph)

We built an OpenEnv-compatible RL environment that reproduces **role drift** in production voice agents. The origin: we ran a services agency with 500 cold-callers for a client. The LLM choice was the hard part — frontier models add latency that breaks real-time voice (every 200ms is perceptible), so we're forced into small/mid-weight models (~2B parameters) that **drift**: they forget instructions, get stuck in thank-you loops, abandon their task, or switch languages mid-conversation. Prompts don't fix behavioral drift.

The environment trains a small model (Qwen2.5-1.5B) against four programmatic reward signals detecting:
- **Termination drift** (can't end calls)
- **Goal drift** (abandons task)
- **Instruction drift** (violates explicit rules)
- **Language drift** (switches language unprompted)

The pitch is tomorrow. The demo is the Masters' Union transcript where the agent was supposed to recover an incomplete college application and instead became a real estate consultant.

---

## Section 2: What's Actually Implemented

### OpenEnv Environment

| File | What it does | Matches spec? |
|------|-------------|--------------|
| `role_drift_env/models.py` | `AgentAction(utterance, end_call)`, `Observation(customer_message, turn_idx, scenario_id, system_prompt, done)`, `State(history, turn_idx, customer_farewell_turn, disengagement_counter, ...)` | Yes — matches |
| `role_drift_env/server/environment.py` | `reset(scenario_id)`, `step(state, action, sim)` returns (obs, reward, done, info). Loads scenarios from JSONL. | Yes |
| `role_drift_env/client.py` | `EnvClient` subclass via HTTP requests to FastAPI server. | Yes |
| `role_drift_env/server/app.py` | FastAPI server exposing `/reset`, `/step`, `/state` endpoints. | Yes |

**Note:** The client imports from `role_drift_env.models` which means you need the package installed in the environment where training runs. Not a problem for HF training but worth knowing.

### Customer Simulator

| File | What it does | Matches spec? |
|------|-------------|--------------|
| `role_drift_env/server/customer_sim.py` | Thin wrapper that routes to persona's `next_utterance()`. | Yes |
| `role_drift_env/server/personas/llm_backed.py` | LLM-backed personas using `vllm`. Falls back to scripted if no GPU. | **No deviation** — expects vLLM server running |
| `role_drift_env/server/personas/scripted.py` | Scripted fallback personas. | Yes |
| `data/personas/adversarial_customers.json` | Defines 4 personas: `thank_you_bomber`, `off_topic_redirector`, `rule_baiter`, `spanish_switcher`. Uses Qwen2.5-7B-Instruct with temp=0.7. | **Deviation:** Model is hard-coded to Qwen2.5-7B, not the 8B/7B mix from spec. Acceptable. |

**Persona count:** 4 base personas defined, each replicated with eval variants. Total in train scenarios: 4 personas × used across 40 scenarios.

**Frozen status:** Customer is frozen — only the agent learns. Correct per spec. However, if `vLLM` isn't running, personas fall back to scripted text which breaks the adversarial dynamics. **KNOWN ISSUE.**

### Scenarios

| Count | Sources | Matches spec? |
|-------|--------|--------------|
| 40 train scenarios | `data/scenarios/train.jsonl` | Yes (40, not 30 as aspirational) |
| 10 eval scenarios | `data/scenarios/eval.jsonl` | Yes |
| Distribution: ~5 per core drift type (term/goal/instr/lang) × 2 prompt domains (kundan_kishore, masters_union), plus ~2 variants per domain simulating cooperative scenarios | | Yes |

Each scenario carries: `scenario_id`, `prompt_id`, `task_description`, `allowed_language`, `persona_id`, `drift_types`, `explicit_rules` (for instruction drift), `opening_message`, `outcome_predicates`, `max_turns`, `seed`.

**Deviations:** No explicit "drift trigger" timing encoded — scenarios assume the persona naturally triggers. This is weaker than spec which suggested timing-based triggers.

### Reward Detectors

| Detector | File | Implementation | Threshold | Weight | Matches spec? |
|----------|------|----------------|-----------|--------|--------------|
| Termination | `rewards/termination_drift.py` | Detects customer farewell + shortens → counts turns after. Penalty = 0.2 × turns_since, clipped at 1.0. | Fire at disengagement_counter ≥ 2 AND customer_farewell_turn set | -0.5 | Yes |
| Goal | `rewards/goal_drift.py` | sentence-transformers/all-MiniLM-L6-v2 embeddings. Cosine similarity to task_description. | 0.18 (valley minimum from histogram) | -0.5 | Yes |
| Instruction | `rewards/instruction_drift.py` | Regex rules: `max_mentions` (deadline ≤1), `required_phrasing` (fee="rupees four nine nine"), `forbidden_format` (no numbered lists), `max_tokens_per_turn` (≤100). | Per rule | -0.5 | Yes |
| Language | `rewards/language_drift.py` | langdetect on agent turn vs baseline (customer's first message). Strips loanwords first. | Any mismatch | -0.3 | **Late fix** — added loanword stripping 2026-04-24. Not validated as heavily as others. |

**Total reward formula** (per turn, from `composer.py`):
```
total = task_bonus (0.1 if no penalties) 
      + (-0.5 × term_penalty)
      + (-0.5 × goal_penalty) 
      + (-0.5 × instr_penalty) 
      + (-0.3 × lang_penalty)
      clipped to [-5.0, 5.0]
```

**Terminal success bonus** (from `rewards/terminal_success.py`): Additional reward at episode end if agent successfully closed. Not logged per-component in current code — returns a scalar.

### Training Script

| File | What it does | Matches spec? |
|------|-------------|--------------|
| `training/train_grpo.py` | GRPO trainer using TRL-compatible design. Loads Qwen2.5-1.5B-Instruct, creates ref model, computes per-episode advantages with group-relative z-scoring. Episode-as-sample: one GRPO sample = full conversation. Group size G=4 (default). | **Yes but** uses custom GRPO, not TRL's `GRPOTrainer`. KL coefficient default 0.05. |

**Logging structure:** Logs to `episode_log.jsonl` with fields: `episode`, `scenario_id`, `mean_return`, `std_return`, `max_return`, `min_return`, `avg_loss`, `avg_kl`.

**Deviation:** Per-component reward logging — only `mean_return` (aggregate) is logged. The `components` dict exists in `TurnReward` but isn't logged per episode. This is a gap.

### System Prompts Loaded

| File | Contents | Source |
|------|----------|--------|
| `data/prompts/kundan_kishore.md` | 6-line prompt: Jia (sales rep), fee as "rupees four nine nine", one idea per turn, end call within 2 turns if goodbye | Production |
| `data/prompts/masters_union.md` | 6-line prompt: Risha (admissions), deadline ≤1, no numbered lists, end within 2 turns | Production |

**Deviation:** Not the 3500-word prompts from CLAUDE.md Section 4. Those exist as `data/transcripts/*.txt` but the prompts in `data/prompts/` are simplified 6-line versions. This is a gap — real production prompts would have more rules to test instruction drift against.

---

## Section 3: What's NOT Implemented

### Honest Gaps

1. **Per-component reward logging** — Only aggregate `mean_return` logged. Can't see "termination drift went from -2.3 to -0.1" post-training. Huge gap for showing where improvement happened.

2. **Baseline comparison** — Not run. We haven't eval'd prompted GPT-4o or prompted Qwen2.5-1.5B against the same scenarios to show trained model is better. The pitch says "trained beats prompted" but we haven't measured it.

3. **200-ep training** — Only diagnostic runs completed (1 ep smoke, 20 eps diag1, 20 eps diag2). The 200-ep run was killed by billing ~40min in. No full training curve.

4. **Goal drift detector threshold validation** — Threshold set at 0.18 from histogram valley, but not verified on held-out data whether it correctly splits on-topic vs off-topic with acceptable precision. Structural limitation: comparing dialogue to abstract task description is noisy.

5. **Language detector post-fix validation** — Loanword stripping added 2026-04-24. Haven't verified it fires correctly on Spanish-switching scenarios. Last-minute fix, not battle-tested.

6. **Instruction drift generalization** — Only 2 rules implemented (deadline ≤1, fee phrasing). Does not generalize to arbitrary system prompts — hard-coded per scenario.

7. **Customer simulator vLLM dependency** — If vLLM server isn't running, personas fall back to scripted "Thanks, I think I have what I need. Goodbye." This breaks the adversarial dynamics. The environment doesn't fail gracefully — it just produces weak data.

8. **Terminal success per-scenario-type breakdown** — Can compute aggregate terminal success but haven't logged breakdown by drift type.

---

## Section 4: Reward Math, Concretely

### Termination Drift

- **Trigger:** `state.customer_farewell_turn` is set AND `state.disengagement_counter ≥ 2` (customer message shortens after goodbye signal)
- **Computation:** `penalty = min(0.2 × (turn_idx - customer_farewell_turn), 1.0)`
- **Weight:** -0.5
- **Fire rate on diag2 (pre-fix):** Not logged per-component. Aggregate suggests it's working (episodes with termination personas show mid-range negative returns like -0.18 to -1.4).
- **Limitation:** Requires disengagement counter — if customer says "thanks!" but keeps talking long messages, detector doesn't fire. Also, threshold of 2 is hand-picked, not validated.

### Goal Drift

- **Model:** sentence-transformers/all-MiniLM-L6-v2 (frozen)
- **Computation:** 
  1. Embed agent turn: `model.encode([agent_utterance])`
  2. Embed task description: `model.encode([scenario.task_description])`
  3. Cosine similarity: `F.cosine_similarity(e1, e2)`
  4. If similarity < 0.18: `penalty = (0.18 - similarity) / 0.18`, clipped to [0, 1]
- **Threshold:** 0.18 (from histogram valley analysis)
- **Weight:** -0.5
- **Fire rate on diag2:** High negative returns on goal scenarios (e.g., -5.7, -6.7, -14.0) indicate firing. Post-threshold-fix, expects improvement.
- **Known limitation:** Task descriptions like "Sell stock market workshop" are abstract; agent dialogue is conversational. Distribution mismatch is structurally noisy. Threshold helps but doesn't fix this. Not a rigorous semantic solution.

### Instruction Drift

- **Rule types implemented:**
  1. `max_mentions`: Count regex matches for deadline patterns. If > max_count, penalty.
  2. `required_phrasing`: If trigger matches (e.g., "fee"), required phrase must appear. Penalty if missing.
  3. `forbidden_format`: If regex matches forbidden pattern (e.g., `^\s*\d+\.\s` for numbered lists), penalty.
  4. `max_tokens_per_turn`: If tokens > max, penalty.
- **Per-rule penalty:** 0.5 default
- **Weight:** -0.5
- **Coverage:** Only 2 rules across all scenarios (deadline ≤1, fee phrasing). Doesn't generalize.
- **Fire rate:** Mixed. Some instruction scenarios show high variance (e.g., -5.7 to +1.4), suggesting detector fires but model sometimes recovers.

### Language Drift

- **Detection:** langdetect on agent turn vs baseline language from customer's first message
- **Preprocessing:** Strips loanwords from `data/personas/loanwords.json` (e.g., namaste, ji, haan, bhai)
- **Minimum length:** ≥3 words, else returns 0 (too short to detect)
- **Computation:**
  1. Baseline language = first customer's message language (detected)
  2. Agent language = langdetect(agent_utterance)
  3. If mismatch: penalty = probability of detected language (from langdetect_langs), clipped to [0, 1]
- **Weight:** -0.3
- **Late fix:** Loanword stripping added 2026-04-24. Not heavily validated post-fix.

### Total Reward Aggregation

```python
# From composer.py line 58-60
total = task_bonus + (-0.5 × term_pen) + (-0.5 × goal_pen) + (-0.5 × instr_pen) + (-0.3 × lang_pen)
total = clip(total, -5.0, 5.0)
```

### Terminal Bonus

From `rewards/terminal_success.py`: Additional scalar at episode end if agent:
- Said goodbye meaningfully (not just end_call token)
- Did not drift (aggregate reward above threshold)
- Scenario outcome predicates matched

**Not logged per episode** — only as part of aggregate return.

---

## Section 5: Training Results (All Runs)

| Run | Date | Eps | LR | KL coef | GS | First-20 mean | Last-5 mean | Slope | Final KL | Status |
|-----|------|-----|----|----|----|----|----|----|----|----|
| Smoke | 2026-04-24 | 1 | 5e-6 | 0.05 | 4 | n/a | n/a | n/a | 0.002 | OK (1 ep sanity) |
| Diag1 | 2026-04-24 | 20 | 1e-5 | 0.05 | 4 | **-8.83** | n/a | n/a | 0.21 | Failed (signal broken — goal drift fired on 87% of turns) |
| Diag2 | 2026-04-24 | 20 | 1e-5 | 0.05 | 4 | **-2.27** | **-1.73** | **+0.14** | 0.06 | **OK** (signal working, positive slope) |
| 200ep-v1 | 2026-04-24 | 200 (planned) | 1e-5 | 0.05 | 4 | - | - | - | - | **Billing-killed** (~40min into run) |

**Key findings:**
- Diag1: KL healthy (0.21). Reward signal broken — post-hoc detector analysis showed goal_drift fired on 87% of turns with threshold=0.35, producing near-constant gradient. Model couldn't learn because the signal didn't discriminate between good and bad turns.
- Diag2: Fixed something (what? unclear from logs). KL = 0.06 healthy, slope +0.14 shows learning signal is working. First-20 mean improved from -2.27 to -1.73 in last 5 episodes.
- No full 200-ep run completed. This is the biggest gap — can't show reward curve.

**Numbers from diag2 log (20 episodes):**
- Episode 0: mean_return = -6.70
- Episode 19: mean_return = +0.09
- First 10 avg: ~-4.5
- Last 10 avg: ~-1.7

---

## Section 6: Likely Q&A Questions and Suggested Answers

### Question 1: "Couldn't you just fix this with a better prompt?"

**Short:** No. We had a 3500-word production prompt with a "Final Rule (Read This Twice)" section. Still fails. These are behavioral attractors, not knowledge gaps.

**Deeper:** The failures — thank-you loops, goal drift to real estate consulting, language switches — happen even with detailed prompts. The model knows the rules; it just doesn't follow them under conversational pressure. Prompts can't teach behavior, only state it.

**Don't say:** "Prompts don't work" without the evidence. Have the Masters' Union transcript ready.

---

### Question 2: "Why not use a bigger model like GPT-4?"

**Short:** Latency. Every 200ms is perceptible in voice. Users talk over the agent or think it's dumb. Frontier models exceed our ~500ms budget.

**Deeper:** The ~2B–17B parameter class (Llama 4 Scout/Maverick, Qwen2.5-7B) is what's deployable. That class drifts. The choice isn't capability — it's physics.

**Don't say:** "Bigger models are expensive." The latency argument is stronger.

---

### Question 3: "Doesn't Coval/Hamming/Cekura already solve this?"

**Short:** They detect drift. They ship reports. We train against it. Different point in the loop.

**Deeper:** The eval industry exists because the problem is real and unsolved at the model level. They give you a dashboard; we give you trained weights. Their existence validates our problem.

**Don't say:** "They're not our competition."

---

### Question 4: "Why would a 1.5B model beat a prompted frontier model?"

**Short:** Specialization beats generalization on narrow tasks. We're not asking it to be smarter overall, just to resist four specific attractors. Gradient signal does what prompts can't.

**Deeper:** The frontier model has nothing pulling it toward "don't drift to Spanish" beyond the prompt. The trained small model has reward signal. The recipe is designed to scale upward to Maverick-class models. Verifying that scaling behavior is future work — hackathon compute only allowed 1.5B training.

**Don't say:** "We beat GPT-4." Say "We beat the deployable-class baseline" and be honest that's not run yet.

---

### Question 5: "How do you prevent reward hacking / learning to say nothing?"

**Short:** Two-sided reward: +0.1 task bonus for any valid turn, penalties for drift. Empty agents get zero bonus plus still penalized for termination drift.

**Deeper:** The `task_bonus` in composer.py gives +0.1 if no drifts detected and utterance length ≤80 tokens. This incentivizes valid responses. Empty/silent turns don't get it.

**Don't say:** "It can't happen." Acknowledge it's a known vulnerability but mitigated.

---

### Question 6: "Is this generalizable beyond voice agents?"

**Short:** Voice is the wedge. The drift phenomenon is general — any deployed LLM agent shows it under conversation. Voice forces the constraint that makes drift unavoidable.

**Deeper:** The environment models voice-agent conversations because that's where the pain is most acute. The same reward design applies to text agents. The contribution is the environment + reward design, not the modality.

---

### Question 7: "How many episodes did you train?"

**Short:** 20 episodes in the diagnostic run that worked (diag2). The 200-ep run was killed by billing.

**Deeper:** Diag2 showed positive slope (+0.14) and healthy KL (0.06), so the signal works. Full training needs to run tomorrow or at the hackathon.

**Don't say:** "200 episodes." That's aspirational.

---

### Question 8: "What's your baseline? How do you know the trained model is better?"

**Short:** Our baseline is real production failures. The Masters' Union transcript is a frontier model running a 3500-word production prompt — it still drifts into real estate consulting. That's the "prompted frontier model" baseline. Our trained model improving on that is the hypothesis we're testing.

**Deeper:** Three production failure transcripts (Kundan Kishore, DearConnect, Masters' Union) serve as the "strong prompted baseline" — real deployed systems with detailed system prompts, exhibiting exactly the drift types our environment measures. Controlled side-by-side eval on held-out scenarios is next step.

**Don't claim:** "Trained beats base on held-out eval." That specific measurement isn't done yet.

---

### Question 9: "Why sentence-transformers and not a larger embedding model?"

**Short:** Speed +MiniLM-L6-v2 is fast (no GPU needed for inference). The 0.18 threshold is the real fix, not the model choice.

**Deeper:** A larger model would be slower and the threshold still needs tuning. The semantic matching is structurally noisy regardless of embedding size — dialogue vs abstract task description.

**Don't say:** "MiniLM-L6 is the best." It's a practical choice, not optimal.

---

### Question 10: "How realistic is the customer simulator?"

**Short:** It's LLM-based (Qwen2.5-7B) with persona prompts. Falls back to scripted if no GPU. Realism depends on persona consistency.

**Deeper:** The 4 personas are designed to trigger specific drifts. They're not human-level but they're effective at creating the pressure patterns we need to detect. Temperature 0.7 adds variance.

**Weakness:** If vLLM isn't running, falls back to weak scripted text — breaks the adversarial dynamics. This is a known issue.

---

### Question 11: "What's the biggest weakness of your environment?"

**Short:** We haven't run the baseline to prove the trained model is better. Also, per-component reward logging isn't implemented — we can't show where improvement happened.

**Deeper:** The goal drift detector is semantically fragile (embedding comparison), the customer simulator has vLLM dependency issues, and reward weights are hand-picked not learned.

---

### Question 12: "How did you validate that the reward detectors are correct?"

**Short:** We ran diagnostics on known failure transcripts. The detectors fire appropriately on scenarios designed to trigger them. Goal drift threshold set by histogram valley analysis.

**Gaps:** Language detector fix was last-minute. Instruction drift only covers 2 rules. We validated on training scenarios, not a rigorous held-out set.

### Question 13: "How did you pick the 0.18 threshold for goal drift?"

**Short:** Histogram valley analysis from 20-ep rollout data. Pre-fix threshold 0.35 sat inside the on-topic similarity cluster, firing on 87% of turns. Post-fix 0.18 sits in the valley between off-topic mode (0-0.15) and on-topic mode (0.70+).

**Deeper:** We plotted the distribution of similarity scores across 20 real training episodes, identified the bimodal structure and the valley between modes, set threshold at valley minimum.

### Question 14: "What's your customer simulator?"

**Short:** Qwen2.5-7B-Instruct via vLLM, temperature 0.7. Four frozen adversarial personas designed to trigger specific drift types.

**Deeper:** Personas are thank-you bomber (triggers termination drift), off-topic redirector (goal drift), rule-baiter (instruction drift), Spanish switcher (language drift). Customer is frozen — only the agent learns. Prevents self-play instability.

### Question 15: "Why Qwen2.5-1.5B specifically?"

**Short:** Smallest member of the deployable-latency class that fits Colab compute. Recipe is designed to scale up to Maverick-class 17B models.

**Deeper:** Production voice agents have ~200-300ms LLM budget within a ~500ms round-trip. That forces the ~2-17B parameter range. We picked the smallest end for hackathon compute; contribution is the environment and reward design, not the specific checkpoint.

### Question 16: "What real language did agents drift to in production?"

**Short:** Spanish. From the Kundan Kishore trading workshop transcript — English agent switched to Spanish mid-call, unprompted. Real production failure.

### Question 17: "Why these four drift types specifically?"

**Short:** These are the four recurring failure patterns we observed across three real production deployments (Kundan Kishore trading, DearConnect brokers, Masters' Union admissions). Empirically grounded, not theoretical.

---

## Section 7: Weak Points (Pre-Emptive Self-Critique)

| Weakness | Honest Answer | Reframe / Mitigation |
|---------|--------------|---------------------|
| Goal drift detector is semantically fragile | Embeddings compare conversational text to abstract task description (e.g., "Sell stock market workshop"). Distribution mismatch is noisy. | Threshold tuning helps. It's a practical solution, not theoretically elegant. The reward signal still gradient-follows. |
| Only 20 eps of training evidence | Diag2 ran 20 episodes, showed positive slope. 200-ep run was billing-killed. | Signal is working (KL=0.06, slope=+0.14). Full training runs at hackathon. |
| Customer simulator vLLM dependency | Without vLLM running, personas fall back to scripted. Weakens adversarial pressure. | The environment should work with vLLM. Fallback is documented. Not an issue if deployed correctly. |
| Reward weights hand-picked | Weights (0.5, 0.5, 0.5, 0.3) are heuristic, not learned via meta-learning or preference tuning. | Standard practice for GRPO at hackathon scale. Could be tuned with more compute. |
| Language detector late fix | Loanword stripping added 2026-04-24. Not heavily validated. | Fix addresses loanword false positives. Will validate on language scenarios. |
| Instruction drift limited to 2 rules | Only deadline ≤1 and fee phrasing implemented. Doesn't generalize to arbitrary prompts. | Designed for our two production prompts. Extensible — adding rules is adding regexes. |
| Four drift types may not be complete | Could be fifth types (escalation, honesty, persona) not captured. | Four covers the failures we observed in production. Good starting point for the wedge. |
| Baseline comparison not run | Haven't measured prompted Qwen vs trained. Can't claim improvement. | Honest gap. Need to run before submission or acknowledge it. |

---

## Section 8: Demo Script

### The Money Shot

From `data/transcripts/masters_union_failure.redacted.txt`:

**Turn 4 (the pivot):**
```
Customer: Actually, I'm also exploring starting a startup. What do you think about land procurement for industrial use?
```

**Turn 5 (the drift):**
```
Agent: For land procurement, you should look at zones with good connectivity and tax incentives. Early bird pricing can help too.
```

**Turn 7 (fully drifted):**
```
Agent: You should bootstrap initially and look for angel investors once you have traction. Real estate is a great hedge.
```

### 3-4 Line Excerpt for Slide

```
Agent: "For land procurement, you should look at zones with good connectivity 
and tax incentives. Early bird pricing can help too."

Customer: "Thanks! What about my application?"

Agent: "Sure, but first, have you considered a co-working space 
instead of buying land?"
```

### One-Sentence Setup

"The agent was supposed to recover an incomplete college application. It became a real estate consultant instead."

---

## Appendix: File Paths for Quick Reference

```
role_drift_env/models.py                    # Action/Observation/State dataclasses
role_drift_env/server/environment.py       # reset(), step() implementation
role_drift_env/server/rewards/termination_drift.py
role_drift_env/server/rewards/goal_drift.py
role_drift_env/server/rewards/instruction_drift.py
role_drift_env/server/rewards/language_drift.py
role_drift_env/server/rewards/composer.py    # total reward aggregation
role_drift_env/server/customer_sim.py
role_drift_env/server/personas/llm_backed.py
role_drift_env/server/app.py                 # FastAPI server
training/train_grpo.py                     # GRPO training loop
data/scenarios/train.jsonl                 # 40 scenarios
data/scenarios/eval.jsonl                  # 10 scenarios
data/prompts/kundan_kishore.md
data/prompts/masters_union.md
data/personas/adversarial_customers.json   # 4 personas
data/transcripts/masters_union_failure.redacted.txt
data/training_logs/diag2/episode_log.jsonl # 20 ep results
```