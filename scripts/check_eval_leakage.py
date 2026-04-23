"""Check eval/train persona overlap (Risk B from review).
If eval reuses the same persona templates as train, eval is in-distribution
and the baseline comparison is softer than claimed.
"""
import json
from collections import Counter

def load_scenarios(path):
    scenarios = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            scenarios.append(json.loads(line))
    return scenarios

train = load_scenarios("data/scenarios/train.jsonl")
eval_set = load_scenarios("data/scenarios/eval.jsonl")

train_personas = Counter(s["persona_id"] for s in train)
eval_personas = Counter(s["persona_id"] for s in eval_set)

print("TRAIN persona distribution:")
for pid, count in train_personas.most_common():
    print(f"  {pid}: {count}")

print("\nEVAL persona distribution:")
for pid, count in eval_personas.most_common():
    print(f"  {pid}: {count}")

# Check overlap
overlap = set(train_personas.keys()) & set(eval_personas.keys())
print(f"\nOverlapping personas: {overlap}")
print(f"Train-only: {set(train_personas.keys()) - set(eval_personas.keys())}")
print(f"Eval-only: {set(eval_personas.keys()) - set(eval_personas.keys())}")

# Check (prompt_id, persona_id) pair overlap
train_pairs = set((s["prompt_id"], s["persona_id"]) for s in train)
eval_pairs = set((s["prompt_id"], s["persona_id"]) for s in eval_set)

pair_overlap = train_pairs & eval_pairs
print(f"\nOverlapping (prompt, persona) pairs: {len(pair_overlap)}")
for pair in pair_overlap:
    print(f"  {pair}")

if pair_overlap:
    print("\nWARNING: Eval scenarios reuse (prompt, persona) combos seen in training.")
    print("This makes eval in-distribution. Consider held-out persona variants.")
else:
    print("\nOK: No (prompt, persona) overlap between train and eval.")
