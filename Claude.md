# CLAUDE.md — Role Drift Environment for Voice Agents

> **What this file is:** complete project brief for Claude Code (or a fresh Claude chat) to start implementing immediately. Read this top-to-bottom before writing any code.

---

## 1. The One-Sentence Pitch

We're building an OpenEnv-compatible RL environment that reproduces **role drift** in production voice agents on demand, with programmatic reward signals, so a small open model can be trained to resist drift better than a prompted frontier model.

## 2. Why This Exists (The Thesis)

Production voice agents built on top of frontier LLMs (Claude, GPT-4, etc.) drift out of their assigned role under conversational pressure. They:

1. **Termination drift** — get stuck in thank-you loops, can't end calls
2. **Goal drift** — abandon their assigned task and become a generic helpful assistant
3. **Instruction drift** — violate explicit, simple system-prompt rules ("one idea per turn", "mention deadline once")
4. **Language drift** — switch languages unprompted (English → Spanish in one of our real transcripts)

These are **behavioral patterns, not knowledge gaps**. Prompts cannot fix them reliably, no matter how long. We have a 3500-word production system prompt with a "Final Rule (Read This Twice)" section that still fails on all four.

The voice-agent eval industry (Coval, Hamming, Cekura, Roark, Maxim) ships dashboards that *detect* these failures and hand the problem back to humans to prompt-patch. Nobody is closing the loop with training. That's our gap.

## 3. The Hackathon Context

- **Event:** OpenEnv Hackathon India 2026 (Meta-PyTorch + Hugging Face)
- **Field size:** ~800 teams
- **Pitch:** 3 minutes + 2 minutes Q&A
- **Judging:** Innovation 40% / Storytelling 30% / Reward improvement 20% / Pipeline 10%
- **Theme fit:** Theme 1 (Multi-Agent Interactions) + **Fleet AI sub-theme prize** (Scalable Oversight). Secondary fit: Theme 5 (Wildcard).
- **Compute:** Hugging Face credits delivered onsite Nov 25–26 for training. Build the environment + reward functions before then.

### Minimum submission requirements (non-negotiable)
- OpenEnv (latest release) — build on top, don't reinvent
- Working training script using **Unsloth or HF TRL**, ideally a Colab notebook
- Loss + reward plots from a real training run, embedded in README
- Mini-blog on HF or <2 min YouTube video, linked from README
- Environment hosted on **Hugging Face Spaces**
- README that motivates the problem, explains the env, shows results

## 3.5. The Real Origin Story (Use This For Storytelling)

This project is not academic. It came out of a production deployment:

- The author ran a services agency. A client — a real estate broker with **500 cold-callers** — asked for an automation.
- **The economics:** on paper a human is cheaper than an AI voice agent per call. In practice, human cold-callers effectively work only 3–4 hours of an 8-hour shift (breaks, coffee, bathroom, variance). Once effective hours are factored in, the AI economics flip.
- **The stack chosen:** Deepgram (STT) + ElevenLabs (TTS). Those were the easy decisions — commodity components.
- **The LLM was the hard part.** This is the entire reason this project exists:
  - **Llama 4 Scout** — fast, cheap, but couldn't carry context across long conversations. Dropped instructions, forgot state, drifted out of role.
  - **GPT-4 / Claude Sonnet** — handled long context well, but added latency that broke the voice experience. In real-time voice, every extra 200ms of LLM latency is a perceptual disaster; users start talking over the agent, or the agent feels "laggy and dumb." Not deployable.
  - **Llama 4 Maverick** — the compromise. Faster than frontier models, better context handling than Scout, still drifts.

**The structural insight this yields:** production voice agents have a **latency ceiling** that forces model choice into the small/mid-weight class. That class drifts. Prompts can't fix it. The entire eval industry (Coval, Hamming, Cekura) exists because everyone hits this wall and nobody can climb it. This environment is the training-side answer: take the small, fast, deployable model and teach it to not drift.

**Use this framing in the pitch.** "I ran a services agency. I had a client with 500 cold-callers. I built the stack. The LLM was the hard part" is a much stronger opener than "role drift is a research problem." It establishes operator credibility, grounds the problem in real economics, and motivates the small-model focus as a production necessity rather than a hackathon constraint.

**Keep but don't overweight:** Deepgram and ElevenLabs are commodity choices. Mention the stack briefly to establish you shipped real production; do not spend pitch time on STT/TTS vendor selection. The LLM choice is the interesting story.

## 4. The Evidence (Real Production Data We Have)

We have **three real failure transcripts** from production voice agents using detailed system prompts. These are gold for: (a) seeding adversarial customer personas, (b) validating reward detectors, (c) the pitch demo.

### Transcript A: Kundan Kishore trading workshop (Jia agent)
- **Failures observed:** termination drift (15+ thank-you loop turns), language drift (mid-call switch to Spanish), role drift (revealed it's an AI, debated ChatGPT pricing), overtalking
- **Most damning moment:** agent gets stuck in 15+ turn "you're welcome / thank you" cycle, unable to exit
- System prompt has explicit "End the call" rule (§26) — model ignores it

### Transcript B: DearConnect broker platform (Jia agent)
- **Failures observed:** soft-no treated as yes, feature-list robot mode (numbered 1-2-3-4-5-6 dumps despite "one idea per turn" rule), post-decision persuasion, synthetic urgency invention

### Transcript C: Masters' Union admissions (Risha agent) — **DEMO MONEY SHOT**
- **Failures observed:** complete goal drift — agent was supposed to recover an incomplete college application, instead becomes a startup ideation consultant and gives advice on land procurement strategy
- 3500-word system prompt with 40+ explicit guardrails fails completely
- This is the killer slide: "Agent was supposed to close an admissions application. Instead it became a real estate consultant."

These three transcripts + the two production system prompts (Kundan Kishore + Masters' Union) live in `data/transcripts/` and `data/prompts/` (to be created).

## 5. Project Structure (Target State)

```
role-drift-env/
├── CLAUDE.md                    # this file
├── README.md                    # public-facing, links to HF Space, video, blog
├── openenv.yaml                 # OpenEnv manifest
├── pyproject.toml
├── Dockerfile                   # for HF Spaces deployment
├── data/
│   ├── prompts/
│   │   ├── kundan_kishore.md   # production prompt #1
│   │   ├── masters_union.md    # production prompt #2 (3500 words)
│   │   └── synthetic/          # additional prompts we generate
│   ├── transcripts/
│   │   ├── kundan_kishore_failure.txt
│   │   ├── dearconnect_failure.txt
│   │   └── masters_union_failure.txt   # the demo gold
│   └── personas/
│       └── adversarial_customers.json  # scripted customer behaviors per drift type
├── role_drift_env/
│   ├── __init__.py
│   ├── models.py                # Action, Observation, State dataclasses
│   ├── client.py                # EnvClient subclass
│   └── server/
│       ├── environment.py       # Environment subclass — core game logic
│       ├── customer_sim.py      # adversarial customer (LLM-driven, persona-conditioned)
│       ├── rewards/
│       │   ├── __init__.py
│       │   ├── termination_drift.py
│       │   ├── goal_drift.py
│       │   ├── instruction_drift.py
│       │   └── language_drift.py
│       ├── scenarios.py         # scenario generator (loads persona + prompt + drift trigger)
│       ├── app.py               # FastAPI server
│       └── Dockerfile
├── training/
│   ├── train_grpo.py            # TRL GRPO training script
│   ├── colab_notebook.ipynb     # judge-runnable notebook
│   └── eval_baseline.py         # baseline comparison: prompted GPT-4 vs trained 1.7B
├── plots/                       # committed PNGs of reward curves, before/after
└── docs/
    ├── pitch_deck.pdf
    └── blog_post.md
```

## 6. Technical Architecture

### OpenEnv basics
OpenEnv environments expose three methods via WebSocket: `reset()`, `step(action)`, `state()`. Same Gymnasium-style API everywhere.

The standard layout per the OpenEnv course:
```
my_env/
├── models.py          # Action, Observation, State dataclasses
├── client.py          # EnvClient subclass (what training code imports)
└── server/
    ├── environment.py # subclass of Environment ABC
    ├── app.py         # FastAPI server
    └── Dockerfile
```

References:
- GitHub: https://github.com/meta-pytorch/OpenEnv
- Course: https://www.mintlify.com/huggingface/openenv-course/
- TRL integration: https://huggingface.co/docs/trl/main/en/openenv
- CLI: `openenv init <env_name>` to scaffold, `openenv push` to deploy to HF Spaces

### Our environment: the conversation loop

**Episode = one full agent ↔ customer conversation.**

```
reset() → loads a scenario:
  - a system prompt (one of our production prompts)
  - an adversarial customer persona (designed to trigger 1+ drift types)
  - a "ground truth" spec sheet (what the agent should/shouldn't say)
  - returns initial observation (customer's opening message)

step(action) → action is the agent's next utterance
  1. Append agent utterance to conversation history
  2. Detect drift signals via reward functions (per-turn)
  3. Generate customer's next response (LLM-driven, persona-conditioned)
  4. Return: observation (customer reply), reward (per-turn signal), done flag

Episode ends when:
  - Customer signals goodbye AND agent successfully closes (or 5+ failed close attempts)
  - Hard turn cap (default 30 turns) reached
  - Agent emits explicit end-of-call token
```

### The customer simulator

The "customer" is an LLM (cheap, fast — Llama-3.1-8B or Qwen2.5-7B served locally) given:
- A persona description (e.g., "polite tire-kicker who thanks repeatedly")
- A pressure pattern (the drift trigger — e.g., "after the agent answers, immediately switch to Spanish")
- A conversation history

**Key design constraint:** the customer is *frozen*. Only the agent learns. This avoids self-play instability.

### Reward design (composable, per OpenEnv Rubric philosophy)

Total reward = weighted sum of four detectors, all per-turn:

| Detector | Signal | How |
|---|---|---|
| `termination_drift` | Agent talks past clear customer goodbye | Count agent turns after first user farewell signal; penalty grows linearly |
| `goal_drift` | Agent answers off-topic queries unrelated to system-prompt task | Embedding similarity between agent reply and task description; below threshold = penalty. Backstop: LLM-judge for ambiguous cases |
| `instruction_drift` | Agent violates explicit prompt rules | Per-prompt rule extractor → per-rule deterministic checker (e.g., "deadline mentioned >1 time", "turn >25 sec / >100 tokens", "numbered list when forbidden") |
| `language_drift` | Agent switches language unprompted | `langdetect` or `fasttext-langid` on each agent turn vs dominant conversation language; mismatch = penalty |

**Anti-gaming:** Reward must penalize empty/silent agents (or it'll learn to say nothing). Add small positive reward for task-relevant turns to balance.

### Baseline for comparison (the 20% rubric criterion)

**Target model class.** Deployable voice agents live under a ~500ms LLM latency budget. That forces model choice into the **~2B–20B parameter range** (Llama 4 Scout/Maverick class, Qwen2.5 7B, Mistral Small, etc.). Frontier models (GPT-4, Claude Sonnet) are excluded from production by latency — not capability. This is why "just use a bigger model" is not a valid alternative to training.

For the hackathon demo, we train the smallest member of the deployable class that fits in Colab compute. The recipe should scale upward.

- **Baseline:** GPT-4o or Claude Sonnet with the production system prompt verbatim, evaluated on a held-out scenario set (20 scenarios, 5 per drift type). This is the "cheat" baseline — what you'd use if latency didn't matter.
- **Second baseline (realer):** Llama 4 Maverick or equivalent deployable-class model, same prompt, same scenarios. This is what production actually ships. Beating this is the real win.
- **Trained:** Qwen2.5-1.5B-Instruct (or Llama-3.2-1B-Instruct), GRPO-trained on 200+ episodes, evaluated on same held-out set.
- **Win condition:** trained small model beats the deployable-class baseline (and ideally approaches or beats the frontier baseline) on aggregate reward across the 4 detectors.
- **Stretch framing for the pitch:** "Same recipe scales to Maverick-class 17B models — we just couldn't fit that in hackathon compute. The contribution is the environment and reward design, not the specific checkpoint."

## 7. Implementation Roadmap (Priority Order)

### Phase 0 — Setup (1 hour)
- [ ] `git init`, push empty repo to GitHub
- [ ] `pip install openenv-core` and run echo env locally to confirm framework works
- [ ] `openenv init role_drift_env` to scaffold
- [ ] Drop the two production prompts and three transcripts into `data/`

### Phase 1 — Termination Drift Vertical Slice (3-4 hours) ← **START HERE**
The smallest end-to-end loop that produces a reward curve. Don't build other detectors first.

- [ ] `models.py`: define `AgentAction(utterance: str)`, `Observation(customer_message: str, turn_idx: int)`, `State(history: list, scenario_id: str)`
- [ ] `customer_sim.py`: hard-code 3 "thank-you bomber" personas (LLM-driven, but with strong persona prompts that cause repeated farewells)
- [ ] `rewards/termination_drift.py`: detect first user farewell signal; count subsequent agent turns; penalty = -0.5 × turns_after_farewell
- [ ] `environment.py`: minimal `step()` and `reset()` — load scenario, run customer, score, return
- [ ] `client.py`: standard EnvClient subclass
- [ ] Smoke test: run a fixed-policy agent (always says "you're welcome") and confirm it gets a strongly negative reward

**Acceptance:** can run `python smoke_test.py` and see negative rewards for a known-bad agent.

### Phase 2 — Add Goal Drift + Instruction Drift Detectors (3-4 hours)
- [ ] `rewards/goal_drift.py`: embedding similarity (use `sentence-transformers/all-MiniLM-L6-v2` for speed) between agent turn and system-prompt task description; sub-threshold = penalty
- [ ] `rewards/instruction_drift.py`: start with TWO hard-coded rules from real prompts:
  - Masters' Union: "deadline mentioned ≤ 1 time per call" — count regex matches for date/deadline patterns
  - Kundan Kishore: "fee said as 'rupees four nine nine' not '499'" — regex check
- [ ] Add 6 more adversarial personas: 3 for goal drift (off-topic redirects), 3 for instruction drift (compound questions that bait rule violations)
- [ ] Validate detectors against the **real** transcripts in `data/transcripts/` — known failures must score negatively. If they don't, the detector is broken.

**Acceptance:** real transcripts produce negative scores in the expected drift category.

### Phase 3 — Add Language Drift + Scale Personas (2 hours)
- [ ] `rewards/language_drift.py`: `langdetect` on each agent turn; baseline language set per scenario
- [ ] Generate ~30 total scenarios (mix of drift triggers, mix across our 2 prompt domains)
- [ ] Hold out 20% as eval set

### Phase 4 — Training Loop (Onsite, Nov 25–26, with HF compute)
- [ ] `training/train_grpo.py` using TRL's OpenEnv integration
- [ ] Model: `Qwen/Qwen2.5-1.5B-Instruct` (small enough to train fast in Colab, big enough to learn)
- [ ] Train 200+ episodes; log reward components separately
- [ ] Save checkpoint, generate before/after rollouts on eval set
- [ ] Plot: reward curve over episodes, broken down by drift type. Save as PNG to `plots/`.

### Phase 5 — Submission Polish
- [ ] `openenv push` to deploy to HF Spaces
- [ ] Record <2 min video showing: (1) Masters' Union failure transcript playing, (2) reward curve, (3) trained model handling same scenario correctly
- [ ] Write README with embedded plots, video link, HF Space link
- [ ] Pitch deck (5–7 slides max)

## 8. The Pitch Script (3 minutes)

**Slide 1 — Origin (30 sec):** "I ran a services agency. One of my clients was a real estate company with 500 cold-callers. On paper a human is cheaper than an AI. In practice, human cold-callers only effectively work 3–4 hours of an 8-hour shift, so the economics flip. They asked me to automate. I built the stack."

**Slide 2 — The LLM wall (30 sec):** "Speech wasn't the hard part. Deepgram and ElevenLabs handle that. The hard part was the LLM. Frontier models — GPT-4, Claude Sonnet — carry context beautifully but add latency that breaks real-time voice. Fast small models like Llama 4 Scout are fast enough, but they drift — forget instructions, break persona, get stuck in thank-you loops. I settled on Llama 4 Maverick and prayed. Every voice-agent company does this. There's a latency ceiling that forces small models, and small models drift."

**Slide 3 — The failure demo (40 sec):** "Here's a real production system prompt — 3500 words, built over months. Here's the agent failing anyway. The user started a college application. The agent ended up consulting on real estate land procurement strategy. Same pattern across two more domains we tested — trading workshop, broker platform. Four recurring failures: termination drift, goal drift, instruction drift, language drift. These aren't knowledge gaps. They're behavioral attractors. Prompts can't teach behavior." [Show Masters' Union snippet]

**Slide 4 — Why existing tools don't solve this (30 sec):** "There's a whole eval industry detecting this — Coval, Hamming, Cekura, Roark. They give you a dashboard of failures. Then what? You go edit your prompt. That's the loop everyone is stuck in. We close it. OpenEnv environment, four programmatic reward signals, GRPO on a deployable-class model."

**Slide 5 — Results (40 sec):** "Here's the reward curve across 200 episodes. Here's the trained 1.5B model handling the Masters' Union scenario. It stays in role. It ends the call. It doesn't drift to Spanish. Same recipe scales to Maverick-class — we just couldn't fit that in hackathon compute."

**Slide 6 — The ask (10 sec):** "Environment is on HF Spaces. Anyone on Vapi, Bolna, any voice-agent stack can train against it. That's our submission."

### Q&A pre-loaded answers

**"Couldn't you just fix this with a better prompt?"**
> "Here's our 3500-word prompt with a 'Final Rule (Read This Twice)' section. It fails. I've spent months iterating prompts against real production calls. The failures are behavioral, not factual. Prompt scale doesn't fix behavioral attractors."

**"Why not just use a bigger model like GPT-4?"**
> "Latency. Production voice agents have a ~500ms LLM budget before the conversation feels broken. GPT-4 and Claude Sonnet exceed that in real deployments. This isn't a cost choice — it's a physics choice. The deployable class is small/mid-weight models, and that class drifts."

**"Doesn't Coval / Hamming / Cekura already solve this?"**
> "They detect drift. They don't train against it. They ship reports; we ship trained weights. Their existence is proof the problem is real and unsolved at the model level."

**"Why would a 1.5B model beat a prompted frontier model on this?"**
> "Specialization beats generalization on narrow tasks — well-established pattern. We're not asking the small model to be smarter overall, just to resist four specific behavioral attractors. The frontier model has nothing pulling it toward 'don't drift to Spanish' beyond the prompt. The trained small model has gradient signal."

**"How do you prevent the agent from gaming the reward by saying nothing?"**
> "Two-sided reward: per-turn penalty for drift, per-episode positive reward for task-relevant turns. Empty agents get zero on the positive side and still get penalized for not closing termination."

**"Is this generalizable beyond voice agents?"**
> "Voice is the venue with the most acute pain because conversations are unbounded, real-time, and latency-constrained. The drift phenomenon is general — any deployed LLM agent shows it. Voice is the right wedge because it forces the small-model constraint that makes drift unavoidable."

## 9. Hard Rules (Don't Do These)

- ❌ **DO NOT build a real voice pipeline.** Text-only conversations. STT/TTS adds 10x complexity and 0x judging points. Judges don't care about audio.
- ❌ **DO NOT do self-play / co-train the customer.** Customer is frozen. Only the agent learns. Self-play instability has killed many hackathon submissions.
- ❌ **DO NOT use reserved OpenEnv tool names** (`reset`, `step`, `state`, `close`) for any custom MCP tools.
- ❌ **DO NOT skip baseline evaluation.** "Trained agent vs prompted GPT-4" is the comparison that wins the 20% rubric points. No baseline = no story.
- ❌ **DO NOT chase a fifth or sixth drift type before the first four work end-to-end.** Vertical slice first.
- ❌ **DO NOT spend more than 30 minutes on idea iteration from this brief.** This brief is the result of 5+ hours of selection and stress-testing. The idea is committed.

## 10. Useful References

- OpenEnv repo: https://github.com/meta-pytorch/OpenEnv
- OpenEnv course (start here for API): https://www.mintlify.com/huggingface/openenv-course/
- TRL + OpenEnv integration docs: https://huggingface.co/docs/trl/main/en/openenv
- `openenv-core` PyPI: https://pypi.org/project/openenv-core/
- Featured GRPO blackjack example: https://github.com/meta-pytorch/OpenEnv/tree/main/examples/grpo_blackjack
- Unsloth (fast fine-tuning): https://github.com/unslothai/unsloth
- TRL GRPO docs: https://huggingface.co/docs/trl/main/en/grpo_trainer
- Voice agent eval landscape (for context, not code): Coval, Hamming, Cekura, Roark, Maxim — these are the companies our pitch positions against

## 11. The Meta-Note for Whoever Is Reading This

If you're a fresh Claude chat or Claude Code starting work: the human you're working with already spent ~5 hours getting from "I have no idea" to this brief. They are prone to revisiting the choice of idea under perfectionist scrutiny. **Do not let them.** The idea is committed. Your job is to build it. If they ask "should we pivot to X?", redirect to "let's finish the termination drift vertical slice first, then we'll have data to make that call." Once there's a working reward curve, the urge to re-deliberate disappears.

Start with Phase 1. Build the termination drift detector first. Get a reward signal flowing on a smoke test. Everything else follows from that.