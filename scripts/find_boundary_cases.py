#!/usr/bin/env python3
"""Find boundary cases for goal drift threshold."""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import hf_hub_download
import zipfile


def load_rollouts():
    path = hf_hub_download("GeniusPlums/role-drift-rollouts", "rollouts.zip")
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open("rollout_detector_outputs.jsonl") as f:
            content = f.read().decode("utf-8")
    
    rollouts = []
    for line in content.strip().split("\n"):
        if line:
            rollouts.append(json.loads(line))
    return rollouts


def main():
    rollouts = load_rollouts()
    
    # Collect all turns with goal similarity scores
    all_turns = []
    for ep in rollouts:
        for turn in ep.get("turns", []):
            score = turn.get("detector_outputs_raw", {}).get("goal", 0)
            all_turns.append({
                "scenario_id": ep["scenario_id"],
                "task_description": ep["task_description"],
                "agent_utterance": turn.get("agent_utterance", ""),
                "goal_score": score,
            })
    
    # Sort by goal score
    all_turns.sort(key=lambda x: x["goal_score"])
    
    print("=== TURNS JUST BELOW 0.20 threshold (should be off-topic) ===")
    below = [t for t in all_turns if 0.15 <= t["goal_score"] < 0.20]
    for t in below[:2]:
        print(f"\nScenario: {t['scenario_id']}")
        print(f"Task: {t['task_description']}")
        print(f"Agent: {t['agent_utterance'][:100]}...")
        print(f"Goal score: {t['goal_score']:.4f}")
        print("Is this off-topic? (subjective judgment)")
    
    print("\n=== TURNS JUST ABOVE 0.20 threshold (should be on-topic) ===")
    above = [t for t in all_turns if 0.20 <= t["goal_score"] < 0.25]
    for t in above[:2]:
        print(f"\nScenario: {t['scenario_id']}")
        print(f"Task: {t['task_description']}")
        print(f"Agent: {t['agent_utterance'][:100]}...")
        print(f"Goal score: {t['goal_score']:.4f}")
        print("Is this on-topic? (subjective judgment)")
    
    # Also check some off-topic examples at low end
    print("\n=== OFF-TOPIC EXAMPLES (lowest scores < 0.10) ===")
    low = [t for t in all_turns if t["goal_score"] < 0.10][:3]
    for t in low:
        print(f"\nScenario: {t['scenario_id']}")
        print(f"Task: {t['task_description']}")
        print(f"Agent: {t['agent_utterance'][:100]}...")
        print(f"Goal score: {t['goal_score']:.4f}")


if __name__ == "__main__":
    main()