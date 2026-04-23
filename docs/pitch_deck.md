# Pitch Deck: Role Drift Environment

## Slide 1: Origin (30 sec)

"I ran a services agency. One of my clients was a real estate company with **500 cold-callers**. On paper a human is cheaper than an AI. In practice, human cold-callers only effectively work 3–4 hours of an 8-hour shift, so the economics flip. They asked me to automate. I built the stack."

---

## Slide 2: The LLM Wall (30 sec)

"Speech wasn't the hard part. Deepgram and ElevenLabs handle that. The hard part was the LLM. Frontier models — GPT-4, Claude Sonnet — carry context beautifully but add latency that breaks real-time voice. Fast small models like Llama 4 Scout are fast enough, but they drift — forget instructions, break persona, get stuck in thank-you loops. I settled on Llama 4 Maverick and prayed. Every voice-agent company does this. There's a latency ceiling that forces small models, and small models drift."

---

## Slide 3: The Failure Demo (40 sec)

"Here's a real production system prompt — 3500 words, built over months. Here's the agent failing anyway. The user started a college application. The agent ended up consulting on real estate land procurement strategy. Same pattern across two more domains we tested — trading workshop, broker platform. Four recurring failures: termination drift, goal drift, instruction drift, language drift. These aren't knowledge gaps. They're behavioral attractors. Prompts can't teach behavior."

**Demo transcript:** `data/transcripts/masters_union_failure.redacted.txt`

---

## Slide 4: Why Existing Tools Don't Solve This (30 sec)

"There's a whole eval industry detecting this — Coval, Hamming, Cekura, Roark. They give you a dashboard of failures. Then what? You go edit your prompt. That's the loop everyone is stuck in. We close it. OpenEnv environment, four programmatic reward signals, GRPO on a deployable-class model."

---

## Slide 5: Results (40 sec)

"Here's the reward curve across 200 episodes. Here's the trained 1.5B model handling the Masters' Union scenario. It stays in role. It ends the call. It doesn't drift to Spanish. Same recipe scales to Maverick-class — we just couldn't fit that in hackathon compute."

**Plots:** `plots/reward_curve.png`, `plots/eval_comparison.png`

---

## Slide 6: The Ask (10 sec)

"Environment is on HF Spaces. Anyone on Vapi, Bolna, any voice-agent stack can train against it. That's our submission."

---

## Q&A Pre-loaded

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
