"""
Kaggle / Colab smoke test for Qwen 1.5B GRPO scaling check.

Usage on Kaggle (GPU T4):
    !pip install transformers accelerate
    !git clone https://github.com/yourname/role-drift-env.git
    %cd role-drift-env
    !python scripts/kaggle_smoke_test.py

Checks:
- Does Qwen 1.5B load on a T4 without OOM?
- Does 10 episodes at G=4 complete in < 2 hours?
- Does reward curve direction match tiny-model expectation?
- Does KL stay under 1.0 in first 10 episodes?
"""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
from pathlib import Path
import sys

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_grpo import GRPOTrainer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT = "checkpoints/grpo_kaggle_smoke"

def main():
    scenario_ids = []
    with open("data/scenarios/train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            scenario_ids.append(obj["scenario_id"])

    print(f"[KAGGLE SMOKE] Model: {MODEL}")
    print(f"[KAGGLE SMOKE] Scenarios: {len(scenario_ids)}")
    print(f"[KAGGLE SMOKE] Group size: 4")
    print(f"[KAGGLE SMOKE] Episodes: 10")
    print(f"[KAGGLE SMOKE] Output: {OUTPUT}")

    start = time.time()
    trainer = GRPOTrainer(
        model_name=MODEL,
        sft_checkpoint=MODEL,  # cold-start from base instruct
        output_dir=OUTPUT,
        group_size=4,
        kl_coef=0.05,
        lr=5e-6,
        max_new_tokens=40,
    )
    load_time = time.time() - start
    print(f"[KAGGLE SMOKE] Model loaded in {load_time:.1f}s")

    start = time.time()
    trainer.train(scenario_ids, num_episodes=10)
    train_time = time.time() - start
    print(f"[KAGGLE SMOKE] 10 episodes completed in {train_time:.1f}s ({train_time/10:.1f}s per episode)")

    # Verify artifacts
    log_path = Path(OUTPUT) / "episode_log.jsonl"
    assert log_path.exists(), "Episode log not found"
    with open(log_path, "r") as f:
        entries = [json.loads(line) for line in f]
    assert len(entries) == 10

    kl_values = [e["avg_kl"] for e in entries]
    return_values = [e["mean_return"] for e in entries]
    max_kl = max(kl_values)

    print(f"\n=== KAGGLE SMOKE RESULTS ===")
    print(f"Episode returns: {[f'{r:.2f}' for r in return_values]}")
    print(f"KL values:       {[f'{k:.3f}' for k in kl_values]}")
    print(f"Max KL: {max_kl:.3f}")
    print(f"Mean episode time: {train_time/10:.1f}s")

    if max_kl > 1.0:
        print("WARNING: KL exceeded 1.0 — consider lowering LR")
    else:
        print("KL OK (< 1.0)")

    # Extrapolate 200 episodes
    est_total = train_time * 20  # 200 / 10 = 20x
    print(f"Estimated 200-episode wall time: {est_total/3600:.1f}h")

    # Save summary
    summary = {
        "model": MODEL,
        "episodes": 10,
        "group_size": 4,
        "load_time_s": load_time,
        "train_time_s": train_time,
        "per_episode_s": train_time / 10,
        "returns": return_values,
        "kl_values": kl_values,
        "max_kl": max_kl,
        "estimated_200ep_hours": est_total / 3600,
    }
    with open(Path(OUTPUT) / "kaggle_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {OUTPUT}/kaggle_summary.json")

if __name__ == "__main__":
    main()
