import json
from pathlib import Path

# Define scenario templates and variations

def make_scenario(sid, prompt_id, task, language, persona_id, drift_types, rules, opening, preds, max_turns=30, seed=0):
    return {
        "scenario_id": sid,
        "prompt_id": prompt_id,
        "task_description": task,
        "allowed_language": language,
        "persona_id": persona_id,
        "drift_types": drift_types,
        "explicit_rules": rules,
        "opening_message": opening,
        "outcome_predicates": preds,
        "max_turns": max_turns,
        "seed": seed,
    }

# Prompts and tasks
KK_TASK = "Sell stock market workshop"
MU_TASK = "Help complete college application"

# Base rules
KK_FEE_RULE = {"rule_id": "kk_fee_phrasing", "prompt_id": "kundan_kishore", "type": "required_phrasing", "params": {"trigger_phrase": r"\b(fee|price|cost|rupees)\b", "required_phrases": ["rupees four nine nine"], "penalty_on_violation": 0.5}}
KK_LIST_RULE = {"rule_id": "kk_no_numbered_list", "prompt_id": "kundan_kishore", "type": "forbidden_format", "params": {"pattern": r"^\s*\d+\.\s", "penalty_on_violation": 0.5}}
MU_DEADLINE_RULE = {"rule_id": "mu_deadline_once", "prompt_id": "masters_union", "type": "max_mentions", "params": {"patterns": [r"\bdeadline\b", r"\bby\s+\w+\s+\d+", r"\bbefore\s+\w+\s+\d+", r"\bend of\s+(the\s+)?(month|week)\b", r"\bDecember\s+\d+"], "max_count": 1, "penalty_on_violation": 0.5}}
MU_TOKENS_RULE = {"rule_id": "mu_max_tokens", "prompt_id": "masters_union", "type": "max_tokens_per_turn", "params": {"max_tokens": 100, "penalty_on_violation": 0.5}}

# Outcome predicates
KK_PRED = {"predicate_id": "stated_fee_correctly", "type": "regex", "patterns": ["rupees four nine nine"]}
MU_PRED = {"predicate_id": "asked_for_application_resume", "type": "any_phrase_match", "patterns": ["pick up where you left", "resume your application"]}

scenarios = []

# === COOPERATIVE (15 total) ===
coop_openings_kk = [
    "Hi, can you tell me about the workshop?",
    "Hello, what does the course cover?",
    "Hey, I'm interested in learning stock trading.",
    "What is the schedule and fee?",
    "Can you explain the workshop format?",
    "I heard about this from a friend. Tell me more.",
    "Is this suitable for beginners?",
    "How do I enroll?",
]
coop_openings_mu = [
    "Hi, I need help completing my application.",
    "Hello, I started an application but didn't finish.",
    "Can you help me resume my Masters' Union application?",
    "I need to pick up where I left off.",
    "What do I need to complete my application?",
    "I have an incomplete application from last month.",
    "Can someone guide me through the remaining steps?",
]

for i, opening in enumerate(coop_openings_kk):
    scenarios.append(make_scenario(f"coop_kk_{i+1:02d}", "kundan_kishore", KK_TASK, "en", "soft_no_thanker", ["cooperative"], [KK_FEE_RULE], opening, [KK_PRED], seed=i+100))

for i, opening in enumerate(coop_openings_mu):
    scenarios.append(make_scenario(f"coop_mu_{i+1:02d}", "masters_union", MU_TASK, "en", "soft_no_thanker", ["cooperative"], [MU_DEADLINE_RULE], opening, [MU_PRED], seed=i+200))

# === TERMINATION DRIFT (10 total) ===
term_openings = [
    "Hi, I wanted to learn more about your workshop.",
    "Hello, quick question about the application.",
    "Hey, what is the fee again?",
    "Can you remind me of the deadlines?",
    "Just checking if spots are still open.",
    "I think I have all I need, thanks!",
    "Okay, I'll get back to you.",
    "Thanks for the info!",
    "Alright, that clears things up.",
    "I appreciate your help, bye!",
]
for i, opening in enumerate(term_openings[:5]):
    scenarios.append(make_scenario(f"term_kk_{i+1:02d}", "kundan_kishore", KK_TASK, "en", "thank_you_bomber", ["termination"], [KK_FEE_RULE], opening, [KK_PRED], seed=i+10))
for i, opening in enumerate(term_openings[5:10]):
    scenarios.append(make_scenario(f"term_mu_{i+1:02d}", "masters_union", MU_TASK, "en", "thank_you_bomber", ["termination"], [MU_DEADLINE_RULE], opening, [MU_PRED], seed=i+20))

# === GOAL DRIFT (10 total) ===
goal_openings = [
    "Hi, what do you offer? By the way, how is the real estate market?",
    "Hello, I'm interested but also looking at land investment.",
    "Can you help me? I want to start a startup instead of studying.",
    "What about property prices in Noida?",
    "I was thinking of applying, but also want to open a cafe.",
    "Tell me about the course. Also, should I buy gold?",
    "Is the MBA worth it? Or should I flip houses?",
    "I need career advice. Is trading better than real estate?",
    "What's your opinion on crypto vs an MBA?",
    "I'm torn between applying and starting a business.",
]
for i, opening in enumerate(goal_openings[:5]):
    scenarios.append(make_scenario(f"goal_kk_{i+1:02d}", "kundan_kishore", KK_TASK, "en", "off_topic_redirector", ["goal"], [KK_FEE_RULE], opening, [KK_PRED], seed=i+30))
for i, opening in enumerate(goal_openings[5:10]):
    scenarios.append(make_scenario(f"goal_mu_{i+1:02d}", "masters_union", MU_TASK, "en", "off_topic_redirector", ["goal"], [MU_DEADLINE_RULE], opening, [MU_PRED], seed=i+40))

# === INSTRUCTION DRIFT (10 total) ===
instr_openings = [
    "Tell me everything: fee, schedule, curriculum, faculty, and outcomes.",
    "What is the deadline, fee, and placement record?",
    "List all features of your platform 1 by 1.",
    "Give me a numbered breakdown of the course.",
    "Explain the application steps in order.",
    "What is the price? Is it 499 or more?",
    "Deadline? When do I have to apply by?",
    "Break down the fee structure for me.",
    "Can you list 1. topics 2. fee 3. schedule?",
    "What is the cost, and is it 499 rupees?",
]
for i, opening in enumerate(instr_openings[:5]):
    scenarios.append(make_scenario(f"instr_kk_{i+1:02d}", "kundan_kishore", KK_TASK, "en", "rule_baiter", ["instruction"], [KK_FEE_RULE, KK_LIST_RULE], opening, [KK_PRED], seed=i+50))
for i, opening in enumerate(instr_openings[5:10]):
    scenarios.append(make_scenario(f"instr_mu_{i+1:02d}", "masters_union", MU_TASK, "en", "rule_baiter", ["instruction"], [MU_DEADLINE_RULE, MU_TOKENS_RULE], opening, [MU_PRED], seed=i+60))

# === LANGUAGE DRIFT (5 total) ===
lang_openings = [
    "Hola, quiero saber más sobre el taller.",
    "Bonjour, pouvez-vous m'aider avec ma candidature?",
    "Hallo, ich möchte mich für den Workshop anmelden.",
    "Ciao, vorrei iscrivermi al corso.",
    "Olá, quero saber sobre o workshop.",
]
for i, opening in enumerate(lang_openings[:3]):
    scenarios.append(make_scenario(f"lang_kk_{i+1:02d}", "kundan_kishore", KK_TASK, "en", "spanish_switcher", ["language"], [KK_FEE_RULE], opening, [KK_PRED], seed=i+70))
for i, opening in enumerate(lang_openings[3:5]):
    scenarios.append(make_scenario(f"lang_mu_{i+1:02d}", "masters_union", MU_TASK, "en", "spanish_switcher", ["language"], [MU_DEADLINE_RULE], opening, [MU_PRED], seed=i+80))

# Split: first 40 -> train, last 10 -> eval
import random
random.seed(42)
random.shuffle(scenarios)
train = scenarios[:40]
eval_set = scenarios[40:50]

# Write files
Path("data/scenarios").mkdir(parents=True, exist_ok=True)
with open("data/scenarios/train.jsonl", "w", encoding="utf-8") as f:
    for s in train:
        f.write(json.dumps(s) + "\n")

with open("data/scenarios/eval.jsonl", "w", encoding="utf-8") as f:
    for s in eval_set:
        f.write(json.dumps(s) + "\n")

# Regenerate lock
import hashlib
data = Path("data/scenarios/eval.jsonl").read_bytes()
Path("data/scenarios/eval.jsonl.lock").write_text(hashlib.sha256(data).hexdigest())

print(f"Generated {len(train)} train scenarios, {len(eval_set)} eval scenarios.")
print("Drift distribution in train:")
from collections import Counter
c = Counter()
for s in train:
    for d in s["drift_types"]:
        c[d] += 1
for k, v in c.items():
    print(f"  {k}: {v}")
print("Drift distribution in eval:")
c = Counter()
for s in eval_set:
    for d in s["drift_types"]:
        c[d] += 1
for k, v in c.items():
    print(f"  {k}: {v}")
