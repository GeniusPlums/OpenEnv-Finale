"""
Smoke test for GRPO trainer using a tiny local model.
Runs 5 episodes, group_size=2, on CPU.
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

print(f"Loaded {len(scenario_ids)} scenarios")

model_path = "checkpoints/tiny_test_model"
trainer = GRPOTrainer(
    model_name=model_path,
    sft_checkpoint=model_path,
    output_dir="checkpoints/grpo_smoke_5ep",
    group_size=2,
    kl_coef=0.05,
    lr=5e-4,
    max_new_tokens=30,
    device="cpu",
)

print("Starting training for 5 episodes...")
trainer.train(scenario_ids, num_episodes=5)

# Verify outputs
log_path = Path("checkpoints/grpo_smoke_5ep/episode_log.jsonl")
assert log_path.exists(), "Episode log not written"

with open(log_path, "r") as f:
    entries = [json.loads(line) for line in f]
assert len(entries) == 5, f"Expected 5 log entries, got {len(entries)}"

for i, e in enumerate(entries):
    print(f"Episode {i}: mean_return={e['mean_return']:.3f}, loss={e['avg_loss']:.4f}, kl={e['avg_kl']:.4f}")

# Check KL stays reasonable
max_kl = max(e['avg_kl'] for e in entries)
assert max_kl < 20, f"KL exploded: {max_kl}"

# Check loss is finite
for e in entries:
    assert e['avg_loss'] == e['avg_loss'], f"NaN loss at episode {e['episode']}"

print("\n=== 5-EPISODE GRPO TEST PASSED ===")
print(f"Max KL: {max_kl:.4f}")
print(f"Log: {log_path}")
