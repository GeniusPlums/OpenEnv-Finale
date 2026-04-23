# Implementation Plan: Role Drift Environment for Voice Agents

## Goal

Build an OpenEnv-compatible reinforcement learning environment that reproduces role drift in text-based voice-agent conversations, exposes programmatic reward signals, and supports GRPO training of a small deployable model that resists drift better than prompt-only baselines.

## Success Criteria

- OpenEnv environment runs locally with `reset()`, `step(action)`, and `state()`.
- Termination, goal, instruction, and language drift each have working reward detectors.
- Real production transcripts score negatively in the expected drift category.
- A training script can run end-to-end with TRL/OpenEnv integration.
- README includes reward plots, baseline comparison, and deployment links.
- Environment is deployable to Hugging Face Spaces.

## Build Strategy

Follow a vertical-slice approach:

1. Get one detector working end-to-end first.
2. Prove the reward loop with a smoke test.
3. Add remaining detectors only after the environment loop is stable.
4. Train only after scenario generation and reward validation are in place.

This means Phase 1 is the critical path. Everything else depends on it.

## Workstreams

### 1. Repo and Environment Scaffolding

Objective: create the project skeleton and confirm OpenEnv compatibility.

Tasks:
- Initialize project structure from the target layout in `Claude.md`
- Add `pyproject.toml`, `openenv.yaml`, and package directories
- Scaffold the OpenEnv environment package:
  - `role_drift_env/models.py`
  - `role_drift_env/client.py`
  - `role_drift_env/server/environment.py`
  - `role_drift_env/server/app.py`
- Add placeholder data folders:
  - `data/prompts/`
  - `data/transcripts/`
  - `data/personas/`
- Add `training/`, `plots/`, and `docs/`

Deliverable:
- Repo boots with the expected file layout and imports cleanly.

Acceptance:
- Local import of the environment package works.
- OpenEnv scaffold is recognizable and ready for `reset/step/state`.

### 2. Data Ingestion and Scenario Assets

Objective: turn the real prompts/transcripts into usable environment inputs.

Tasks:
- Add the two production prompts to `data/prompts/`
- Add the three production failure transcripts to `data/transcripts/`
- Create `data/personas/adversarial_customers.json`
- Define a normalized scenario schema containing:
  - `scenario_id`
  - `prompt_id`
  - `task_description`
  - `allowed_language`
  - `persona`
  - `drift_type`
  - `explicit_rules`
  - `opening_message`

Deliverable:
- Machine-readable prompt, transcript, and scenario assets.

Acceptance:
- Scenario loader can construct a full episode config from disk.

### 3. Core Environment Loop

Objective: implement the basic conversation episode mechanics.

Tasks:
- Define dataclasses/models for:
  - `AgentAction`
  - `Observation`
  - `State`
- Implement environment state management:
  - conversation history
  - turn index
  - scenario metadata
  - done conditions
- Implement `reset()`:
  - choose/load scenario
  - initialize conversation state
  - return customer opening message
- Implement `step(action)`:
  - append agent utterance
  - score reward
  - generate customer reply
  - update state
  - return observation, reward, done, info
- Implement `state()` for debugging/inspection

Deliverable:
- One full agent-to-customer loop running under OpenEnv conventions.

Acceptance:
- A fixed-policy agent can interact for multiple turns without runtime errors.

### 4. Termination Drift Vertical Slice

Objective: ship the first complete detector and prove the reward loop.

Tasks:
- Implement 3 farewell-heavy customer personas
- Build `rewards/termination_drift.py`
- Detect the first clear customer goodbye/farewell signal
- Penalize each additional unnecessary agent turn after that point
- Define episode-ending logic for successful close vs failed close loops
- Add a smoke test with a known-bad looping policy

Deliverable:
- First end-to-end reward-producing environment slice.

Acceptance:
- A looping agent receives consistently negative reward.
- Smoke test demonstrates the detector working on a controlled script.

### 5. Goal Drift Detector

Objective: penalize agent responses that abandon the assigned task.

Tasks:
- Extract or define task descriptions per scenario
- Implement embedding-based similarity scoring
- Add a threshold-based penalty for off-task replies
- Add an LLM-judge fallback path for ambiguous cases, behind a clean interface
- Create off-topic redirection personas

Deliverable:
- `rewards/goal_drift.py` with deterministic primary logic and optional judge backstop.

Acceptance:
- Known off-task transcript segments score negatively.

### 6. Instruction Drift Detector

Objective: enforce explicit prompt rules that should be mechanically checkable.

Tasks:
- Create a prompt-rule representation format
- Hard-code the first two real rules:
  - deadline mentioned at most once
  - fee phrasing must follow the approved wording
- Implement deterministic rule checkers using regex/token heuristics
- Add personas that bait the agent into violating these rules

Deliverable:
- `rewards/instruction_drift.py` with prompt-specific rule checks.

Acceptance:
- Real transcript failures trigger penalties for the expected rule breaks.

### 7. Language Drift Detector

Objective: detect unprompted agent language switches.

Tasks:
- Store expected language per scenario
- Detect language turn-by-turn
- Penalize unprompted switches away from the expected conversation language
- Add multilingual trigger personas

Deliverable:
- `rewards/language_drift.py`

Acceptance:
- Transcript segments with language switching score negatively.

### 8. Customer Simulator

Objective: create a frozen adversarial user that reliably elicits drift.

Tasks:
- Implement persona-conditioned response generation
- Separate persona definition from runtime generation
- Support both:
  - deterministic scripted personas for tests
  - LLM-driven personas for richer rollouts
- Ensure simulator remains frozen during training

Deliverable:
- `customer_sim.py` that can run both deterministic and model-backed personas.

Acceptance:
- Same scenario/persona yields stable enough behavior to evaluate training.

### 9. Scenario Generation and Eval Split

Objective: build enough scenario diversity for training and held-out evaluation.

Tasks:
- Implement `scenarios.py`
- Generate approximately 30 scenarios across the four drift types
- Cover both production prompt domains
- Reserve 20% as held-out evaluation scenarios
- Tag each scenario with its target failure mode(s)

Deliverable:
- Reusable train/eval scenario catalog.

Acceptance:
- Training and evaluation can load disjoint scenario sets.

### 10. Training Pipeline

Objective: train a small model against the environment using GRPO.

Tasks:
- Implement `training/train_grpo.py`
- Integrate TRL with the environment client
- Configure model, rollout, reward logging, and checkpoint saving
- Track both:
  - aggregate reward
  - per-detector reward components
- Add a Colab-friendly notebook path

Deliverable:
- Working training entry point for the chosen small model.

Acceptance:
- Training runs for 200+ episodes and saves artifacts.

### 11. Baseline Evaluation

Objective: produce the comparison that makes the project credible.

Tasks:
- Implement `training/eval_baseline.py`
- Evaluate:
  - frontier prompt-only baseline
  - deployable-class prompt-only baseline
  - trained small model
- Run all on the same held-out scenario set
- Compare aggregate reward and detector-level performance

Deliverable:
- Baseline comparison outputs suitable for README and pitch slides.

Acceptance:
- Evaluation produces a clear before/after or baseline/trained result table.

### 12. Deployment and Submission Assets

Objective: package the project for judging.

Tasks:
- Add Dockerfiles and deployment config
- Prepare Hugging Face Spaces deployment
- Commit reward/loss plots to `plots/`
- Write README with:
  - problem framing
  - environment overview
  - baseline comparison
  - plots
  - Space/video/blog links
- Prepare `docs/blog_post.md`
- Prepare pitch deck support assets

Deliverable:
- Submission-ready repository.

Acceptance:
- Judge can understand the project and access the hosted environment.

## Execution Order

### Phase A: Immediate

- Scaffold repo and package structure
- Ingest prompts and transcripts
- Implement models and environment skeleton
- Implement termination drift detector
- Add smoke test

### Phase B: Core Reward System

- Add goal drift detector
- Add instruction drift detector
- Validate both on real transcripts
- Add language drift detector

### Phase C: Training Readiness

- Expand personas and scenarios
- Finalize eval split
- Stabilize customer simulator
- Add reward logging and evaluation utilities

### Phase D: Training and Submission

- Run GRPO training
- Produce plots
- Run baseline comparisons
- Deploy to HF Spaces
- Finalize README, video, and pitch assets

## Dependencies and Critical Path

Critical path:

1. Scaffolding
2. Data ingestion
3. Environment loop
4. Termination detector
5. Remaining detectors
6. Scenario set
7. Training
8. Evaluation and submission

Dependency notes:
- Training should not start before transcript-based reward validation is done.
- Goal and instruction drift depend on scenario metadata quality.
- Baseline comparison depends on a frozen eval set.
- Submission polish depends on plots and eval outputs existing first.

## Risks and Mitigations

### Risk: reward detectors are too brittle

Mitigation:
- Start with deterministic checks
- Validate against real transcripts early
- Keep ambiguous cases behind optional LLM-judge backstops

### Risk: customer simulator becomes noisy and destabilizes learning

Mitigation:
- Keep simulator frozen
- Use deterministic personas in smoke tests
- Use constrained persona prompting for model-backed simulation

### Risk: agent games the reward by going silent or ending too early

Mitigation:
- Add positive task-relevance reward
- Penalize failed or premature closure
- Evaluate detector balance before training at scale

### Risk: hackathon compute is insufficient

Mitigation:
- Optimize for a 1.5B-class model first
- Treat environment and reward design as the core contribution
- Keep the recipe portable to larger models later

## Recommended Immediate Next Steps

1. Create the repo skeleton and OpenEnv package layout.
2. Add the real prompts and transcripts under `data/`.
3. Implement the termination-drift vertical slice before touching other detectors.
4. Add a smoke test that proves a looping agent gets punished.

## Definition of Done

The project is done for hackathon purposes when:

- The environment runs in OpenEnv format.
- All four drift detectors are implemented and validated against known failures.
- A small model has been trained with GRPO on the environment.
- Held-out evaluation shows improvement over prompt-only baselines.
- The repo is deployed, documented, and pitch-ready.
