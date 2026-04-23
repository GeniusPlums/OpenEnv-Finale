"""
Smoke test for GRPO trainer using a tiny local model.
Runs 3 episodes, group_size=2, on CPU.
Verifies: model load, generation, reward flow, advantage compute, 
         policy update, checkpoint save, log write.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from pathlib import Path
from training.train_grpo import GRPOTrainer

# Load scenario IDs
scenario_ids = []
with open("data/scenarios/train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        scenario_ids.append(obj["scenario_id"])

print(f"Loaded {len(scenario_ids)} scenarios")

# Use locally-created tiny model (6.9M params, no download needed)
model_path = "checkpoints/tiny_test_model"
assert Path(model_path).exists(), f"Tiny model not found at {model_path}. Run scripts/create_tiny_model.py first."

print(f"Initializing GRPOTrainer from local tiny model: {model_path}")
trainer = GRPOTrainer(
    model_name=model_path,
    sft_checkpoint=model_path,
    output_dir="checkpoints/grpo_smoke",
    group_size=2,
    kl_coef=0.05,
    lr=5e-4,
    max_new_tokens=30,
    device="cpu",
)

print("Starting training for 3 episodes...")
trainer.train(scenario_ids, num_episodes=3)

# Verify outputs
log_path = Path("checkpoints/grpo_smoke/episode_log.jsonl")
assert log_path.exists(), "Episode log not written"

best_path = Path("checkpoints/grpo_smoke/best")
has_checkpoint = (best_path / "model.safetensors").exists() or (best_path / "pytorch_model.bin").exists()
assert has_checkpoint, "Best checkpoint not saved"

# Verify log structure
with open(log_path, "r") as f:
    entries = [json.loads(line) for line in f]
assert len(entries) == 3, f"Expected 3 log entries, got {len(entries)}"
for i, e in enumerate(entries):
    assert "episode" in e
    assert "mean_return" in e
    assert "std_return" in e
    assert "avg_loss" in e
    assert "avg_kl" in e
    print(f"Episode {i}: mean_return={e['mean_return']:.3f}, loss={e['avg_loss']:.4f}, kl={e['avg_kl']:.4f}")

print("\n=== GRPO SMOKE TEST PASSED ===")
print(f"Log file: {log_path} ({len(entries)} entries)")
print(f"Checkpoint: {best_path}")

# Verify checkpoint can be reloaded
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained(best_path)
t = AutoTokenizer.from_pretrained(best_path)
print(f"Checkpoint reloads OK: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")
