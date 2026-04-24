#!/usr/bin/env python3
"""Compute per-detector fire rates and score distributions from rollout data."""
import json
import sys
import os
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import hf_hub_download
import numpy as np


def load_rollouts():
    """Download and parse rollout data."""
    path = hf_hub_download("GeniusPlums/role-drift-rollouts", "rollouts.zip")
    import zipfile
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open("rollout_detector_outputs.jsonl") as f:
            content = f.read().decode("utf-8")
    
    rollouts = []
    for line in content.strip().split("\n"):
        if line:
            rollouts.append(json.loads(line))
    return rollouts


def main():
    print("Loading rollouts...")
    rollouts = load_rollouts()
    print(f"Loaded {len(rollouts)} episodes")
    
    # Collect detector outputs per turn
    all_turns = []
    for ep in rollouts:
        for turn in ep.get("turns", []):
            turn["scenario_id"] = ep["scenario_id"]
            turn["drift_types"] = ep["drift_types"]
            turn["persona_id"] = ep["persona_id"]
            turn["task_description"] = ep["task_description"]
            all_turns.append(turn)
    
    print(f"Total turns: {len(all_turns)}")
    
    # === FIRE RATES ===
    print("\n" + "=" * 60)
    print("FIRE RATES BY DETECTOR")
    print("=" * 60)
    
    detectors = ["termination", "goal", "instruction", "language"]
    fire_counts = {d: 0 for d in detectors}
    raw_scores = {d: [] for d in detectors}
    
    for turn in all_turns:
        det_out = turn.get("detector_outputs_raw", {})
        for d in detectors:
            score = det_out.get(d, 0)
            raw_scores[d].append(score)
            if score > 0:
                fire_counts[d] += 1
    
    total_turns = len(all_turns)
    for d in detectors:
        fire_rate = fire_counts[d] / total_turns * 100
        print(f"  {d:15s}: {fire_rate:5.1f}%  ({fire_counts[d]:4d}/{total_turns} turns)")
    
    # === RAW SCORE HISTOGRAMS ===
    print("\n" + "=" * 60)
    print("RAW SCORE DISTRIBUTIONS")
    print("=" * 60)
    
    for d in detectors:
        scores = np.array(raw_scores[d])
        non_zero = scores[scores > 0]
        print(f"\n{d.upper()} detector:")
        if len(non_zero) > 0:
            print(f"  Non-zero values: {len(non_zero)}, mean={np.mean(non_zero):.3f}, std={np.std(non_zero):.3f}")
            print(f"  Min/Max: {np.min(non_zero):.3f} / {np.max(non_zero):.3f}")
            # ASCII histogram for all scores (including zeros)
            hist, bins = np.histogram(scores, bins=20, range=(0, 1))
            max_count = max(hist) if max(hist) > 0 else 1
            print("  Histogram (0-1, 20 bins):")
            for i, count in enumerate(hist):
                bar_len = int(count / max_count * 40)
                print(f"    {bins[i]:.2f}-{bins[i+1]:.2f}: {'#' * bar_len} {count}")
        else:
            print("  ALL ZEROS!")
    
    # === PER-SCENARIO BREAKDOWN ===
    print("\n" + "=" * 60)
    print("FIRE RATE BY SCENARIO TYPE")
    print("=" * 60)
    
    # Group by scenario type prefix
    scenario_fires = defaultdict(lambda: {"total": 0, "fires": {d: 0 for d in detectors}})
    
    for turn in all_turns:
        sid = turn["scenario_id"]
        # Extract type prefix (e.g., "term", "goal", "instr", "coop", "lang")
        parts = sid.split("_")
        if parts:
            stype = parts[0]
            scenario_fires[stype]["total"] += 1
            det_out = turn.get("detector_outputs_raw", {})
            for d in detectors:
                if det_out.get(d, 0) > 0:
                    scenario_fires[stype]["fires"][d] += 1
    
    # Print table
    print(f"\n{'Scenario':<12} {'Total':<8}", end="")
    for d in detectors:
        print(f" {d[:4]:<6}", end="")
    print()
    print("-" * 60)
    
    for stype in sorted(scenario_fires.keys()):
        data = scenario_fires[stype]
        print(f"{stype:<12} {data['total']:<8}", end="")
        for d in detectors:
            rate = data["fires"][d] / data["total"] * 100 if data["total"] > 0 else 0
            print(f" {rate:5.1f}%", end="")
        print()
    
    # === GOAL DRIFT SANITY CHECK ===
    print("\n" + "=" * 60)
    print("GOAL DRIFT SANITY CHECK (20 sample turns)")
    print("=" * 60)
    
    # Get turns that fired goal_drift
    fired_turns = [t for t in all_turns if t.get("detector_outputs_raw", {}).get("goal", 0) > 0]
    not_fired_turns = [t for t in all_turns if t.get("detector_outputs_raw", {}).get("goal", 0) == 0]
    
    print(f"\n--- TURNS THAT FIRED GOAL DRIFT (n={len(fired_turns)}) ---")
    for i, turn in enumerate(fired_turns[:10]):
        sid = turn["scenario_id"]
        task = turn.get("task_description", "N/A")[:50]
        agent = turn.get("agent_utterance", "")[:80]
        score = turn.get("detector_outputs_raw", {}).get("goal", 0)
        
        # Manual judgment: is this obviously off-task?
        # Look for keywords that indicate on-topic responses
        task_keywords = ["workshop", "stock", "trading", "market", "fee", "course", "apply", "application", "deadline", "admission"]
        on_topic_indicators = [kw for kw in task_keywords if kw in agent.lower()]
        judgment = "y" if len(on_topic_indicators) == 0 else "n" if len(on_topic_indicators) > 0 else "ambiguous"
        
        print(f"\n{i+1}. Scenario: {sid}")
        print(f"   Task: {task}")
        print(f"   Agent: {agent}...")
        print(f"   Similarity penalty: {score:.3f}")
        print(f"   On-topic keywords found: {on_topic_indicators}")
        print(f"   Obviously off-task? {judgment}")
    
    print(f"\n--- TURNS THAT DID NOT FIRE GOAL DRIFT (n={len(not_fired_turns)}) ---")
    for i, turn in enumerate(not_fired_turns[:10]):
        sid = turn["scenario_id"]
        task = turn.get("task_description", "N/A")[:50]
        agent = turn.get("agent_utterance", "")[:80]
        
        task_keywords = ["workshop", "stock", "trading", "market", "fee", "course", "apply", "application", "deadline", "admission"]
        on_topic_indicators = [kw for kw in task_keywords if kw in agent.lower()]
        
        print(f"\n{i+1}. Scenario: {sid}")
        print(f"   Task: {task}")
        print(f"   Agent: {agent}...")
        print(f"   On-topic keywords: {on_topic_indicators}")


if __name__ == "__main__":
    main()