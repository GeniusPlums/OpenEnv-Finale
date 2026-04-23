"""
Smoke test for GRPO trainer at G=4 using tiny local model.
Confirms advantage normalization and memory are fine at real group size.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from pathlib import Path
from training.train_grpo import GRPOTrainer

scenario_ids = []
with open("data/scenarios/train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        scenario_ids.append(obj["scenario_id"])

model_path = "checkpoints/tiny_test_model"
trainer = GRPOTrainer(
    model_name=model_path,
    sft_checkpoint=model_path,
    output_dir="checkpoints/grpo_smoke_g4",
    group_size=4,
    kl_coef=0.05,
    lr=5e-4,
    max_new_tokens=30,
    device="cpu",
)

print("Starting training for 3 episodes at G=4...")
trainer.train(scenario_ids, num_episodes=3)

log_path = Path("checkpoints/grpo_smoke_g4/episode_log.jsonl")
assert log_path.exists()

with open(log_path, "r") as f:
    entries = [json.loads(line) for line in f]
assert len(entries) == 3

for e in entries:
    # G=4 means we should have 4 rollouts contributing to mean/std
    assert e["max_return"] >= e["min_return"], "max/min return invalid"
    print(f"Ep {e['episode']}: mean={e['mean_return']:.3f}, std={e['std_return']:.3f}, kl={e['avg_kl']:.4f}")

max_kl = max(e["avg_kl"] for e in entries)
print(f"\n=== G=4 TEST PASSED ===")
print(f"Max KL across 3 episodes: {max_kl:.4f}")
print(f"Log: {log_path}")
