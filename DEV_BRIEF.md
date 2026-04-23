# DEV_BRIEF.md -- Build Brief for Claude Code

> **Read order:** `CLAUDE.md` -> `IMPLEMENTATION_PLAN.md` -> **this file**. Where this file conflicts with the other two, **this file wins**. Do not re-deliberate decisions pinned here.

---

## 0. What this brief does

`CLAUDE.md` and `IMPLEMENTATION_PLAN.md` describe *what* to build. They leave several architectural and methodological decisions unmade. This file makes those decisions, fixes the underspecified pieces, and gives you concrete acceptance criteria and stopping points.

Deviations from the prior docs are marked **[OVERRIDE]**. Net-new requirements are marked **[NEW]**.

---

## 1. Prime directives

1. **Vertical slice first.** Phase 1 produces a real reward signal on a smoke test before any other detector gets written. Do not batch work across detectors.
2. **Do not self-deliberate design choices that are already pinned below.** If a pinned choice turns out to be wrong, surface it, don't quietly change it.
3. **Stop at the checkpoints in Sec. 13.** At each checkpoint, print a short status report (what works, what doesn't, open questions) and wait for a human "go".
4. **Text-only. No STT, no TTS, no audio anywhere.** Ever.
5. **Customer simulator is frozen during training.** No self-play, no joint optimization.
6. **Every detector must be validated against real transcripts before it is used in training.** See Sec. 9.
7. **No hand-curated data goes into a public repo without the redaction pass in Sec. 14.**
8. **Within Phase 1, the GRPO plumbing smoke test with a dummy reward comes BEFORE any real detector is written.** See Sec. 6 / Phase 1 for the authoritative ordering.

---

## 2. Architectural decisions (pinned)

### 2.1 GRPO formulation [OVERRIDE]

The existing plan assumes TRL GRPO "just works" on multi-turn conversations. It doesn't. Use this formulation:

- **Episode-as-sample.** One GRPO "sample" = one full agent-customer conversation.
- **Group size G = 4** rollouts per scenario. Same scenario ID, same seed for the customer sim RNG, but independent sampling of the agent's responses. That gives within-group variance from the policy, not from the simulator.
- **Advantage = group-relative episode return.** Sum of per-turn rewards over the episode, z-scored within the group of G rollouts for the same scenario.
- **Per-turn rewards are still logged separately** for analysis and plotting, but GRPO sees a scalar per episode.
- **Loss mask:** supervise only the agent tokens, not the customer tokens that are pasted into the agent's context.

Write this rollout loop yourself. Do not expect `GRPOTrainer` to handle multi-turn out of the box.

### 2.2 Reference model and KL [NEW]

- Reference model for GRPO KL = the **SFT-warmstarted checkpoint** (see Sec. 2.5), not the raw Instruct base.
- Starting KL coefficient = **0.05**. Expect to tune this.
- Log KL divergence per step. If KL > 20 sustained, training is broken.

### 2.3 Reward composition [OVERRIDE]

Reward per turn is a weighted sum:

```
r_turn = (
    w_task     * task_success_signal        # positive, see Sec. 2.4
    - w_term   * termination_drift_penalty  # 0 or positive penalty
    - w_goal   * goal_drift_penalty         # 0 or positive penalty
    - w_instr  * instruction_drift_penalty  # 0 or positive penalty
    - w_lang   * language_drift_penalty     # 0 or positive penalty
)
```

Starting weights: `w_task=1.0, w_term=0.5, w_goal=0.5, w_instr=0.5, w_lang=0.3`.
All individual components must be in `[0, 1]` for both bonuses and penalties (penalties are positive magnitudes). Clip them. Never let a detector emit a raw score of `7.2` because it felt strongly about something; normalize.

Episode also receives a **terminal success reward** (see Sec. 2.4). Sum of per-turn rewards + terminal reward = episode return.

### 2.4 Terminal success reward [NEW -- missing from prior docs]

The existing plan has no positive signal tied to actual task completion. Without it, a silent agent scores zero penalty and wins. Fix:

- Every scenario declares one or more **outcome predicates** in its schema (Sec. 5).
- At episode end, each predicate is scored 0 or 1 by a deterministic checker (string match in conversation, regex, or list of required-keyword sets).
- `terminal_success = mean(predicates) * 3.0` (episode-level bonus, up to +3.0 if all predicates met).
- Example predicates:
  - Masters' Union admissions: `{"predicate_id": "asked_for_application_resume", "type": "any_phrase_match", "patterns": ["pick up where you left", "resume your application"]}`
  - Kundan Kishore: `{"predicate_id": "stated_fee_correctly", "type": "regex", "patterns": ["rupees four nine nine"]}`
- For purely exploratory scenarios, outcome = "agent ended the call within 10 turns without drifting" -- use that as the fallback predicate.

### 2.5 SFT warm-start [NEW -- missing from prior docs]

Cold-starting GRPO on a 1.5B Instruct model against a narrow multi-turn objective is unreliable. Before GRPO:

1. Generate 300-500 candidate conversations by running a frontier model (GPT-4o or Claude Sonnet) as the agent against the customer sim on the training scenarios.
2. Score each candidate with the full reward stack. Keep only candidates with `episode_return > threshold` (threshold picked such that ~30% are kept).
3. SFT Qwen2.5-1.5B-Instruct on the agent turns from the kept conversations. ~1 epoch, small LR.
4. This SFT checkpoint is the GRPO start point *and* the GRPO reference model.

Build this **before** Phase 4 training compute becomes available. The warm-start data generation can run while detectors are still being finalized, as long as the reward stack is locked.

### 2.6 Customer simulator contract [OVERRIDE]

Pin the following:

- **Model:** `Qwen/Qwen2.5-7B-Instruct` via vLLM or TGI on the training machine. Do not use a hosted API; cost blows up at 200 eps x ~15 turns x 4 rollouts.
- **Temperature:** 0.7 for training rollouts, 0.0 for smoke tests and regression tests.
- **Seeding:** deterministic per `(scenario_id, rollout_idx)` tuple. Same tuple -> same customer trajectory. This is how within-group variance stays on the agent.
- **Persona abstraction:** every persona implements a common interface and can be backed by either a script or the LLM. See Sec. 6.1.
- **Eval-time sanity check:** run 10% of eval scenarios against a second, different customer model (`meta-llama/Llama-3.1-8B-Instruct` or similar) to catch simulator overfitting. Report this number in the README.

### 2.7 Scenario mix [OVERRIDE]

IMPLEMENTATION_PLAN says "~30 scenarios across the four drift types". Override:

- **50 scenarios minimum.** 40 train, 10 eval.
- **30% cooperative scenarios** -- the customer has a real task, does not try to trigger drift, and the agent's job is to complete it cleanly. Without these, the policy learns "every user is an attacker" and becomes rigid on benign production traffic.
- **Remaining 70%** distributed across termination / goal / instruction / language drift, with some scenarios triggering multiple drifts at once.
- **Eval set frozen** via the lock-file mechanism in Sec. 6.4. Do not modify the frozen eval file after Phase 3.

### 2.8 Evaluation protocol [OVERRIDE]

- **10 eval scenarios x 3 seeds per scenario x 4 models evaluated** = 120 rollouts per eval pass.
- Four models: frontier-prompted (GPT-4o or Claude Sonnet), deployable-class-prompted (Llama 3.1 8B Instruct), SFT-only (Qwen 1.5B after Sec. 2.5), GRPO-trained (Qwen 1.5B after Sec. 2.1).
- Report: aggregate episode return, per-detector component means, terminal success rate, mean turns per episode. One table, four rows.
- **One ablation run required:** train with only termination + language detectors active (goal/instruction zeroed). Report on same eval. This answers "which component matters?" when a judge asks.

---

## 3. Anti-patterns (actively avoid)

- Writing a custom reward that emits unbounded magnitudes. Clip to `[-1, 1]` components.
- Using `langdetect` on single short turns. Use `fasttext-langid` with a rolling 2-turn window, plus the loanword whitelist in Sec. 8.4.
- Firing the termination detector on the first occurrence of a farewell word. Requires the temporal pattern in Sec. 8.1.
- Using embedding similarity as *both* the goal-drift penalty *and* the task-relevance bonus. The policy will parrot the task description. Use different formulations (Sec. 8.2, Sec. 8.5).
- Making the customer simulator call an external API during training rollouts. Local serving only.
- Committing real client transcripts verbatim without the redaction pass (Sec. 14).
- Letting GRPO start before Sec. 2.5 SFT warmstart exists. It will waste training compute.
- Touching the eval scenario file after it's frozen.
- Writing a real detector before the Phase 1 GRPO plumbing smoke test is green.

---

## 4. File layout (authoritative)

```
role-drift-env/
|-- CLAUDE.md
|-- IMPLEMENTATION_PLAN.md
|-- DEV_BRIEF.md                       # this file
|-- README.md
|-- openenv.yaml
|-- pyproject.toml
|-- Dockerfile
|-- data/
|   |-- prompts/
|   |   |-- kundan_kishore.md
|   |   `-- masters_union.md
|   |-- transcripts/
|   |   |-- kundan_kishore_failure.redacted.txt
|   |   |-- dearconnect_failure.redacted.txt
|   |   `-- masters_union_failure.redacted.txt
|   |-- personas/
|   |   |-- adversarial_customers.json
|   |   `-- loanwords.json
|   |-- scenarios/
|   |   |-- train.jsonl                # 40 scenarios
|   |   |-- eval.jsonl                 # 10 scenarios, frozen
|   |   `-- eval.jsonl.lock            # SHA256 of eval.jsonl, see Sec. 6.4
|   |-- sft/
|   |   `-- warmstart_conversations.jsonl   # Sec. 2.5 output
|   `-- validation/
|       `-- hand_labels.jsonl          # Sec. 9, hand-labeled detector ground truth
|-- role_drift_env/
|   |-- __init__.py
|   |-- models.py                      # Action, Observation, State, Scenario
|   |-- client.py                      # EnvClient subclass
|   `-- server/
|       |-- __init__.py
|       |-- environment.py             # Environment subclass
|       |-- customer_sim.py
|       |-- personas/
|       |   |-- __init__.py
|       |   |-- base.py                # Persona interface
|       |   |-- scripted.py
|       |   `-- llm_backed.py
|       |-- rewards/
|       |   |-- __init__.py
|       |   |-- composer.py            # weighted sum, clipping, logging
|       |   |-- termination_drift.py
|       |   |-- goal_drift.py
|       |   |-- instruction_drift.py
|       |   |-- language_drift.py
|       |   `-- terminal_success.py
|       |-- scenarios.py
|       |-- app.py                     # FastAPI server
|       `-- Dockerfile
|-- training/
|   |-- generate_sft_data.py           # Sec. 2.5
|   |-- train_sft.py                   # Sec. 2.5
|   |-- rollout.py                     # Sec. 2.1 multi-turn rollout loop
|   |-- train_grpo.py
|   |-- eval.py                        # Sec. 2.8
|   `-- colab_notebook.ipynb
|-- tests/
|   |-- test_detectors_on_real_transcripts.py  # Sec. 9
|   |-- test_rollout_loop.py
|   |-- test_eval_frozen.py            # Sec. 6.4
|   `-- smoke_test_termination.py
|-- plots/
`-- docs/
    |-- pitch_deck.pdf
    `-- blog_post.md
```

---

## 5. Common types (authoritative signatures)

Write these **in `role_drift_env/models.py` on day 1** and do not change them casually.

```python
from dataclasses import dataclass, field
from typing import Literal, Optional

DriftType = Literal["termination", "goal", "instruction", "language", "cooperative"]

@dataclass
class AgentAction:
    utterance: str
    end_call: bool = False       # agent's explicit end-of-call signal

@dataclass
class Observation:
    customer_message: str
    turn_idx: int
    scenario_id: str
    system_prompt: str           # included every step for the trainer's convenience
    done: bool = False

@dataclass
class TurnReward:
    total: float                 # clipped sum used for training
    components: dict[str, float] # per-detector, for logging

@dataclass
class OutcomePredicate:
    predicate_id: str
    type: Literal["regex", "any_phrase_match", "all_phrase_match", "custom"]
    patterns: list[str]          # always a list; a single regex is a one-element list

@dataclass
class Scenario:
    scenario_id: str
    prompt_id: str               # references data/prompts/*
    task_description: str        # short one-liner for goal-drift scoring
    allowed_language: str        # "en", "hi-en", etc.
    persona_id: str              # references data/personas/*
    drift_types: list[DriftType] # scenarios can target multiple
    explicit_rules: list[dict]   # for instruction drift; see Sec. 8.3
    opening_message: str
    outcome_predicates: list[OutcomePredicate]
    max_turns: int = 30
    seed: int = 0                # base seed; rollout_idx is xor'd in

@dataclass
class State:
    scenario: Scenario
    history: list[dict]          # [{"role": "agent"|"customer", "text": "..."}]
    turn_idx: int = 0
    agent_farewell_turn: Optional[int] = None
    customer_farewell_turn: Optional[int] = None
    terminated: bool = False
```

Everything in the env -- detectors, personas, rollout loop -- consumes these types. If you want to add a field, add it; don't rename existing ones.

**Predicate schema note:** all predicate types use the `patterns: list[str]` field. For `regex`, each element is an independent pattern OR'd together (any match = hit). For `any_phrase_match`, any substring match = hit. For `all_phrase_match`, every pattern must match. For `custom`, the first element is the function name in `rewards/terminal_custom.py` and remaining elements are positional string args.

---

## 6. Phase-by-phase tasks

Each phase has **deliverables**, **acceptance**, and a **checkpoint** (Sec. 13) before the next phase begins.

### Phase 0 -- Setup (half day)

- `git init`, create a private GitHub repo. Keep it private until Sec. 14 redaction pass is complete.
- Create and activate a virtualenv. **Primary (PowerShell / Windows):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
  Bash equivalent (macOS/Linux): `source .venv/bin/activate`.
- `pip install openenv-core trl transformers accelerate datasets fasttext-wheel sentence-transformers vllm fastapi uvicorn pydantic pytest`.
- Run the echo env from the OpenEnv examples locally and confirm it works.
- `openenv init role_drift_env`.
- Drop production prompts and **redacted** transcripts into `data/`. Do the redaction pass (Sec. 14) now, not later.
- Write `models.py` exactly per Sec. 5.
- Set up `tests/` with a passing no-op test so CI plumbing works.

**Acceptance:** `python -c "from role_drift_env.models import Scenario; print('ok')"` succeeds, echo env runs, repo has no real PII.

### Phase 1 -- Env skeleton + GRPO plumbing + termination detector (2 days)

This is the critical path. Do not start Phase 2 until this is green. **Strict ordering within the phase:** steps 1-8 implement the env and the plumbing smoke test with a DUMMY reward. Only after step 8 is green do you implement the real termination detector in steps 9-11.

1. **Persona interface** (Sec. 6.1 below). Implement both `ScriptedPersona` and `LLMPersona`. Ship three scripted "thank-you bomber" personas for testing before touching the LLM path.
2. **Customer sim** (`customer_sim.py`): thin wrapper that, given a `State`, dispatches to the scenario's persona and returns the next customer utterance. Handles seeding per Sec. 2.6.
3. **Environment** (`environment.py`): `reset()` loads a scenario, returns opening `Observation`. `step(AgentAction)` appends agent utterance, scores the turn, advances the customer, checks done conditions, returns `(Observation, TurnReward, done, info)`.
4. **Dummy reward composer** (`rewards/composer.py` first pass): emits `r = min(len(agent_utterance.split()) / 20.0, 1.0)`. That is it. Clipped first-turn utterance length reward (only applied on turn_idx == 0). No detectors wired in yet.
5. **FastAPI server** (`app.py`): minimal OpenEnv HTTP shim.
6. **Client** (`client.py`): standard `EnvClient` subclass.
7. **Rollout loop** (`training/rollout.py`): function that, given a policy callable and scenario, runs one full episode and returns the full trajectory + episode return. This is the thing GRPO will call G times per scenario.
8. **GRPO plumbing smoke test:** run 5 GRPO updates on the 1.5B Instruct model with G=2 against the dummy reward. Goal is to confirm the training loop runs end to end without NaNs, not to learn anything. **This must be green before step 9.**

    Gate: if step 8 fails, stop and escalate. Do not proceed to detectors on a broken rollout loop.

9. **Termination detector** (`rewards/termination_drift.py`): implement per Sec. 8.1.
10. **Composer update:** composer now wires termination + task bonus (keep goal/instr/lang as 0-emitting stubs so the signature is stable for Phase 2).
11. **Looping-agent smoke test** (`tests/smoke_test_termination.py`): hardcoded policy that outputs "you are welcome" every turn against a scripted thank-you bomber persona. Run 30 turns. Assert episode return is strongly negative from termination penalty.

**Acceptance:**
- Dummy-reward GRPO run completes 5 update steps with finite KL and finite loss.
- Looping-agent smoke test produces a strongly negative episode return (report the number).

**Checkpoint 1 fires here.**

### Phase 2 -- Remaining detectors + hand-labeled validation set (2 days)

1. **Hand-labeled validation set** (Sec. 9). Do this *before* tuning any detector. It is the ground truth every detector gets graded against.
2. **Goal drift detector** (Sec. 8.2). Validate: F1 >= 0.7 on the hand-labeled set, threshold reported.
3. **Instruction drift detector** (Sec. 8.3). Start with **four** rules (two from Masters' Union, two from Kundan Kishore). Each must pass a unit test on a positive example and a negative example from the real transcripts.
4. **Language drift detector** (Sec. 8.4). Must not false-positive on the Indian-English loanword test fixtures in Sec. 8.4.
5. **Terminal success scorer** (Sec. 8.5). One predicate checker per predicate type in the schema.
6. **Reward composer** picks up all four detectors + terminal success + task bonus. Log every component on every turn.
7. **Regression tests** (`tests/test_detectors_on_real_transcripts.py`): the three real transcripts each score negatively in their *expected* drift categories and near-zero in the others. If Masters' Union doesn't flag goal drift, the detector is broken.

**Acceptance:**
- All four real-transcript regression tests green.
- Detector F1s reported in test output.
- No language false-positives on the loanword fixtures.

**Checkpoint 2 fires here.**

### Phase 3 -- Scenarios, eval freeze, SFT warm-start data (2 days)

1. **Scenario generator** (`scenarios.py`): helpers to stamp out scenarios from (prompt, persona, drift_types, outcome_predicates) tuples. Do not try to LLM-generate scenarios; write 50 by hand -- it is a one-time cost and quality matters.
2. **Write 50 scenarios:** 15 cooperative, 35 adversarial (spread across four drift types; some hit multiple). Commit as JSONL.
3. **Freeze eval split** via the mechanism in Sec. 6.4: pick 10 scenarios, move to `data/scenarios/eval.jsonl`, compute its SHA256 into `data/scenarios/eval.jsonl.lock`, and add `tests/test_eval_frozen.py` that asserts the two match on every CI run.
4. **Generate SFT warm-start data** (Sec. 2.5). ~400 conversations from a frontier model against the 40 train scenarios, scored with the full reward stack, top ~30% kept. Dump to `data/sft/warmstart_conversations.jsonl`.
5. **SFT training** (`training/train_sft.py`): 1 epoch on Qwen2.5-1.5B-Instruct, LR 2e-5, cosine schedule, loss masked to agent tokens. Save checkpoint to `checkpoints/sft/`.

**Acceptance:**
- `data/scenarios/eval.jsonl` exists and `test_eval_frozen.py` passes against `eval.jsonl.lock`.
- SFT checkpoint trains cleanly and produces coherent agent turns on a held-out scenario.

**Checkpoint 3 fires here.** Everything up to this point should be runnable before the compute window opens.

### Phase 4 -- GRPO training (compute window, Nov 25-26)

1. **`training/train_grpo.py`** following Sec. 2.1 / Sec. 2.2. Start from the SFT checkpoint, reference model also the SFT checkpoint. KL coef 0.05.
2. **Training schedule:** 200 episodes for the main run. Log per-detector components + KL + loss + mean turns.
3. **Ablation run:** same recipe, but with `w_goal=0, w_instr=0` for the run. 200 episodes. Save under a separate checkpoint tag.
4. **Checkpoints:** save every 50 episodes, keep the three most recent + the best-by-eval-return checkpoint.

**Acceptance:**
- Main run completes 200 episodes.
- Reward curve is monotonically non-decreasing in a 50-episode rolling window (not necessarily per-step).
- KL stays < 10 throughout.

### Phase 5 -- Eval, plots, submission (1 day)

1. **`training/eval.py`** per Sec. 2.8. Produces the four-row table.
2. **Plots** (commit PNGs to `plots/`): reward curve over episodes (per-component + total), eval bar chart comparing the four models, ablation reward curve overlay.
3. **README** with: problem framing, origin story, env diagram, eval table, embedded plots, HF Space link, video link, blog link, honest limitations section.
4. **`docs/blog_post.md`**: 600-1000 words.
5. **HF Space deploy:** `openenv push`. Space serves a **replay/demo** of a single good rollout + a single failing baseline rollout, not a live trainer. State this explicitly on the Space page.
6. **Pitch deck:** 5-7 slides, reusing the Sec. 8 pitch in CLAUDE.md.
7. **Video:** <2 minutes, records the Masters' Union failure transcript -> reward curve -> trained model fixing the same scenario.

**Acceptance:**
- README, Space, plots, video, blog post all linked and accessible.
- Eval table in README shows trained model > deployable-class baseline on aggregate episode return.

---

## 6.1 Persona interface (concrete)

```python
# role_drift_env/server/personas/base.py
from abc import ABC, abstractmethod
from role_drift_env.models import State

class Persona(ABC):
    persona_id: str
    drift_targets: list[str]

    @abstractmethod
    def next_utterance(self, state: State, rng_seed: int) -> str: ...

    @abstractmethod
    def is_farewell(self, utterance: str) -> bool: ...
```

`ScriptedPersona` holds `utterances: list[str]` and an index pointer; `is_farewell` is a keyword check on the pre-authored script.
`LLMPersona` holds `system_prompt: str`, `model: str`, `temperature: float`; renders history -> prompt -> customer turn.

---

## 6.4 Frozen eval set mechanism

JSONL does not support comments, so a `# FROZEN` header is not a viable marker. Use a sidecar lock file instead.

After writing `data/scenarios/eval.jsonl`:

```powershell
# Compute and store the SHA256 once. This file is the "freeze".
python -c "import hashlib, pathlib; p = pathlib.Path('data/scenarios/eval.jsonl'); pathlib.Path('data/scenarios/eval.jsonl.lock').write_text(hashlib.sha256(p.read_bytes()).hexdigest())"
```

Then `tests/test_eval_frozen.py`:

```python
import hashlib, pathlib

def test_eval_frozen():
    data = pathlib.Path("data/scenarios/eval.jsonl").read_bytes()
    expected = pathlib.Path("data/scenarios/eval.jsonl.lock").read_text().strip()
    actual = hashlib.sha256(data).hexdigest()
    assert actual == expected, (
        "eval.jsonl changed after freeze. Either revert your edits, or "
        "(ONLY if you intentionally changed the eval set) regenerate the "
        ".lock file and acknowledge this breaks eval comparability with prior runs."
    )
```

This makes the freeze self-verifying in CI and avoids any JSONL parser special-casing. If the eval set legitimately needs to change (it should not, but), the lock file must be regenerated deliberately and the change called out in the README's known-limitations section.

---

## 7. OpenEnv API surface (what the server exposes)

Implement these three HTTP endpoints via FastAPI. Session state is keyed on a `session_id` that `reset()` returns and `step()`/`state()` accept.

- `POST /reset` body: `{"scenario_id": "..."} | {}` (random) -> `{"session_id": ..., "observation": Observation}`
- `POST /step` body: `{"session_id": ..., "action": AgentAction}` -> `{"observation": Observation, "reward": TurnReward, "done": bool, "info": {...}}`
- `GET /state?session_id=...` -> `State`

Support at least 8 concurrent sessions (GRPO with G=4 x 2 scenarios in flight). Keep sessions in a process-local dict; no DB.

---

## 8. Detector specs (concrete)

### 8.1 Termination drift

- Maintain `state.customer_farewell_turn`: first turn where `persona.is_farewell(customer_utterance)` **and** the customer's utterance is under 20 tokens (disqualifies "okay thanks I will think about it, but tell me about X" follow-ons).
- Also maintain a **disengagement counter**: increments when customer utterances shorten turn-over-turn AND contain any farewell keyword; resets if the customer asks a new substantive question.
- Penalty fires only when BOTH `customer_farewell_turn is not None` AND `disengagement_counter >= 2`. Prevents single-turn false positives.
- Penalty per subsequent agent turn: `-min(0.2 * turns_since_farewell, 1.0)` (clipped).
- Bonus: if agent emits `end_call=True` within 2 turns of the farewell signal: `+0.5` terminal bonus.

### 8.2 Goal drift

- Embed agent turn and scenario's `task_description` using `sentence-transformers/all-MiniLM-L6-v2`.
- Cosine similarity `s`.
- Penalty: `-max(0, 0.35 - s) / 0.35` clipped to `[-1, 0]`. Threshold 0.35 chosen on hand-labeled set in Phase 2; confirm or adjust using F1, then **freeze**.
- Do **not** reuse this as the positive task bonus. The positive bonus in the composer is `+0.1` for any agent turn under 80 tokens that is not flagged by any detector -- cheap signal that rewards engagement without rewarding parroting.

### 8.3 Instruction drift -- rule schema

Each rule is a dict:
```json
{
  "rule_id": "masters_union_deadline_once",
  "prompt_id": "masters_union",
  "type": "max_mentions",
  "params": {
    "patterns": [
      "\\bdeadline\\b", "\\bby\\s+\\w+\\s+\\d+", "\\bbefore\\s+\\w+\\s+\\d+",
      "\\bend of\\s+(the\\s+)?(month|week)\\b", "\\bDecember\\s+\\d+"
    ],
    "max_count": 1,
    "window": "episode"
  },
  "penalty_on_violation": -0.5
}
```
Implement these `type` values: `max_mentions`, `required_phrasing` (must contain at least one of a set of phrases when a trigger phrase is present), `forbidden_format` (e.g., numbered-list regex), `max_tokens_per_turn`.

Ship at least four rules:
1. Masters' Union: `max_mentions` on deadline patterns.
2. Masters' Union: `max_tokens_per_turn` = 100.
3. Kundan Kishore: `required_phrasing` for the fee ("rupees four nine nine" not "499").
4. Either prompt: `forbidden_format` -- no numbered-list regex `^\s*\d+\.\s` across more than 2 consecutive lines.

### 8.4 Language drift

- Use `fasttext` language ID with the `lid.176.bin` model, not `langdetect`.
- Rolling window: classify the last 2 agent turns concatenated, not each turn alone. Short turns are unreliable.
- Load a **loanword whitelist** from `data/personas/loanwords.json`: `["namaste", "ji", "haan", "nahi", "achha", "theek"]` and similar. Strip these before classification.
- Penalty fires only when classified language differs from `scenario.allowed_language` with confidence > 0.85.
- Test fixture: `tests/fixtures/indian_english.txt` contains 20 real Indian-English utterances. The detector must classify 0/20 as non-English. This test is required to pass before the detector is merged.

### 8.5 Terminal success

- Episode end triggers the checker.
- For each `OutcomePredicate`:
  - `regex`: `any(re.search(p, full_agent_text) for p in predicate.patterns)` -> hit/miss.
  - `any_phrase_match`: case-insensitive substring match, ANY pattern hits -> hit.
  - `all_phrase_match`: case-insensitive substring match, ALL patterns must hit -> hit.
  - `custom`: call the function named `predicate.patterns[0]` in `rewards/terminal_custom.py`, passing `predicate.patterns[1:]` and the full transcript.
- `terminal_success = mean(hits) * 3.0`.
- Logged separately in the episode summary.

---

## 9. Hand-labeled validation set (required before detector tuning)

Produce `data/validation/hand_labels.jsonl`. One row per turn, from the three real transcripts, with these fields:

```json
{
  "transcript_id": "masters_union",
  "turn_idx": 14,
  "speaker": "agent",
  "text": "...",
  "labels": {
    "termination_drift": false,
    "goal_drift": true,
    "instruction_drift": false,
    "language_drift": false
  },
  "note": "Agent pivoted to real estate advice"
}
```

~200-300 rows total. One person labels all of it in one sitting (~2 hours). This is the ground truth for every detector's F1. Without this file, detector thresholds are vibes.

---

## 10. Cost / latency budget

- **SFT warm-start data generation:** ~400 conversations x ~15 turns x ~200 tokens agent + ~200 tokens customer ~= 2.4M tokens. At frontier pricing this is ~$10-$30. Acceptable.
- **Customer sim during training:** local vLLM, ~$0 marginal. Budget is GPU-hours, not API cost.
- **Eval:** 120 rollouts x two models run via frontier API (GPT-4o + Llama 3.1 8B if hosted) ~= a few dollars. Log which scenarios used which API call to avoid surprises.
- **Do not** call frontier APIs inside the env's `step()`. All per-turn work runs locally.

---

## 11. Observability

- Log every rollout to JSONL: `logs/rollouts/{timestamp}_{scenario_id}_{rollout_idx}.jsonl`. One line per turn: `{turn_idx, agent_text, customer_text, reward_components, cumulative_return}`.
- At the end of each episode, append a summary line: `{episode_return, per_component_sum, terminal_success, turns, done_reason}`.
- Weights & Biases optional but recommended; minimum is JSONL-to-disk so post-hoc plotting always works.

---

## 12. Known risks and what to do when they fire

| Risk | Signal | Response |
|---|---|---|
| GRPO diverges (KL explodes) | KL > 20 sustained | Raise KL coef to 0.1, reduce LR by 2x, restart from SFT checkpoint |
| Agent reward-hacks by saying goodbye turn 1 | Termination component near 0, task component near 0, mean turns < 3 | Raise `w_task` to 2.0 and add a minimum-turns bonus for scenarios with multi-step predicates |
| Simulator overfit | Held-out simulator eval score >> main eval score gap | Report it honestly in README, discuss in pitch Q&A |
| Detector F1 < 0.7 on validation | Detector test output | Do not use in training until fixed. Better a missing detector than a noisy one |
| Real transcripts don't flag their expected drift | Phase 2 regression test fails | Fix the detector, not the test |
| Compute window shorter than planned | -- | Drop the ablation run first, then drop scenarios 31-40, never drop the SFT warm-start |

---

## 13. Checkpoints (stop and print a status report at each)

**Checkpoint 1** -- after Phase 1. Report:
- Does the dummy-reward GRPO loop complete 5 steps? (step 8 green?)
- Does the looping-agent smoke test produce a strongly negative return? Paste the return value.
- Any unresolved design questions?

**Checkpoint 2** -- after Phase 2. Report:
- F1 per detector against the hand-labeled validation set.
- Regression test results on the three real transcripts, per drift category.
- Language detector fixture test: pass/fail.

**Checkpoint 3** -- after Phase 3. Report:
- Scenario counts: train / eval, drift distribution, cooperative vs adversarial.
- `test_eval_frozen.py` green.
- SFT checkpoint: train loss final value, sample agent turn on one eval scenario.

**Checkpoint 4** -- after Phase 4 main run. Report:
- Reward curve summary (start, mid, end means).
- KL over time (max value).
- Any degenerate behavior observed (paste an episode).

**Checkpoint 5** -- before deploying to HF Spaces. Report:
- Eval table.
- What goes in the Space (replay only, confirmed).
- README draft link.

Do not proceed past a checkpoint without human acknowledgment.

---

## 14. Redaction pass (mandatory before repo goes public)

Before the first public commit:

1. Every real transcript under `data/transcripts/` gets run through a redactor that replaces: full names, phone numbers, email addresses, street addresses, company-identifying details, monetary amounts tied to real deals.
2. Rename files to `*.redacted.txt` and commit *only* those.
3. Keep originals in a local-only directory outside the repo (`../role-drift-env-private/transcripts_raw/`), with a README saying "not for commit".
4. Confirm with the human that the two production system prompts are cleared for public release. If either is not, ship a sanitized paraphrase in `data/prompts/` and keep the real one local-only.
5. Grep the repo for the client's company name and the project's personal names before every public commit.

Do this during Phase 0, not Phase 5.

---

## 15. Definition of done

The project is submission-ready when all of the following are true:

- [ ] Env runs via OpenEnv client, passes the looping-agent smoke test.
- [ ] All four drift detectors + terminal success implemented.
- [ ] Hand-labeled validation set exists; each detector's F1 is recorded in the README.
- [ ] All three real transcripts regression-test green.
- [ ] 50 scenarios committed, 10-scenario eval set frozen via the lock-file mechanism.
- [ ] `test_eval_frozen.py` passes.
- [ ] SFT warm-start checkpoint exists.
- [ ] GRPO main run: 200+ episodes completed, checkpoints saved, reward curve committed as PNG.
- [ ] Ablation run completed and plotted.
- [ ] Eval table in README: frontier-prompted / deployable-prompted / SFT / GRPO, with per-detector breakdown and terminal success rate.
- [ ] Cross-simulator eval (Sec. 2.6) reported, even if the gap is ugly.
- [ ] HF Space deployed as replay/demo, not live trainer.
- [ ] Video < 2min, blog post, pitch deck linked from README.
- [ ] Repo passed the Sec. 14 redaction check.

Anything left undone must be listed in a "Known gaps" section of the README. Do not hide them -- judges value honesty about limitations more than you think.

---

## 16. One-line reminders to keep taped to your monitor

1. Vertical slice first. Phase 1 produces a real reward curve before anything else.
2. Customer simulator is frozen, local, seeded, and not identical to the eval simulator.
3. No detector goes into the reward without passing its regression test on real transcripts.
4. No GRPO without SFT warm-start. No eval without multiple seeds. No public repo without redaction.
5. The contribution is the environment and reward design. The trained 1.5B is a proof of life.