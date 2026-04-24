#!/usr/bin/env python3
"""Diagnostic analysis script for 20-episode GRPO run."""
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import List, Dict
import numpy as np


def load_episode_log(repo_path: str) -> List[dict]:
    """Download and parse episode_log.jsonl from HF Hub."""
    try:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(repo_id=repo_path, filename="episode_log.jsonl")
        episodes = []
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))
        return episodes
    except Exception as e:
        print(f"Error loading episode log: {e}")
        return []


def compute_return_trend(episodes: List[dict]) -> Dict:
    """Compute return trend across episodes."""
    # Each episode has aggregated stats from the group
    returns = [ep["mean_return"] for ep in episodes]
    
    if len(episodes) < 20:
        return {"error": f"Not enough episodes: {len(episodes)}", "returns": returns}
    
    early_mean = mean(returns[:5])
    late_mean = mean(returns[15:])
    delta = late_mean - early_mean
    
    # Simple linear fit
    episodes_nums = list(range(len(returns)))
    coeffs = np.polyfit(episodes_nums, returns, 1)
    slope = coeffs[0]
    
    return {
        "early_mean": early_mean,
        "late_mean": late_mean,
        "delta": delta,
        "slope": slope,
        "early_std": stdev(returns[:5]) if len(returns) >= 5 else 0,
        "late_std": stdev(returns[15:]) if len(returns) >= 15 else 0,
    }


def compute_kl_trajectory(episodes: List[dict]) -> Dict:
    """Compute KL divergence trajectory."""
    # Each episode has aggregated KL from the group
    kls = [ep.get("avg_kl", 0) for ep in episodes]
    
    windows = []
    for i in range(0, len(kls), 5):
        window = kls[i:i+5]
        if window:
            windows.append(mean(window))
    
    initial_kl = mean(kls[:5]) if len(kls) >= 5 else mean(kls)
    final_kl = mean(kls[-5:]) if len(kls) >= 5 else mean(kls)
    
    return {
        "initial_kl": initial_kl,
        "final_kl": final_kl,
        "windows": windows,
        "all_kls": kls,
    }


def compute_scenario_coverage(episodes: List[dict]) -> Dict:
    """Compute scenario coverage."""
    scenario_ids = [ep.get("scenario_id", "unknown") for ep in episodes]
    unique = set(scenario_ids)
    counts = {sid: scenario_ids.count(sid) for sid in unique}
    
    return {
        "unique_count": len(unique),
        "unique_scenarios": list(unique),
        "counts": counts,
    }


def compute_reward_breakdown(episodes: List[dict]) -> Dict:
    """Compute per-component reward breakdown."""
    components = {
        "termination": [],
        "goal": [],
        "instruction": [],
        "language": [],
    }
    
    for ep in episodes:
        rew = ep.get("reward", {})
        if isinstance(rew, dict):
            for key in components:
                components[key].append(rew.get(key, 0))
    
    breakdown = {}
    for key, vals in components.items():
        if vals:
            breakdown[key] = {
                "mean": mean(vals),
                "std": stdev(vals) if len(vals) > 1 else 0,
                "min": min(vals),
                "max": max(vals),
            }
    
    return breakdown


def main():
    if len(sys.argv) < 2:
        repo_path = "GeniusPlums/role-drift-runs-diag"
    else:
        repo_path = sys.argv[1]
    
    print(f"Loading episodes from {repo_path}...")
    episodes = load_episode_log(repo_path)
    
    if not episodes:
        print("ERROR: No episodes found in log")
        sys.exit(1)
    
    print(f"Loaded {len(episodes)} episodes\n")
    
    # A. Return trend
    print("=" * 50)
    print("A. RETURN TREND")
    print("=" * 50)
    trend = compute_return_trend(episodes)
    print(f"  Episodes 0-4 mean return:  {trend.get('early_mean', 'N/A'):.3f} (std {trend.get('early_std', 0):.3f})")
    print(f"  Episodes 15-19 mean return: {trend.get('late_mean', 'N/A'):.3f} (std {trend.get('late_std', 0):.3f})")
    print(f"  Delta (late - early):       {trend.get('delta', 'N/A'):.3f}")
    print(f"  Linear fit slope:           {trend.get('slope', 'N/A'):.4f}")
    
    # B. KL trajectory
    print("\n" + "=" * 50)
    print("B. KL TRAJECTORY")
    print("=" * 50)
    kl = compute_kl_trajectory(episodes)
    print(f"  Initial KL (ep 0-4):  {kl['initial_kl']:.4f}")
    print(f"  Final KL (ep 15-19):  {kl['final_kl']:.4f}")
    print(f"  KL per window (0-4,5-9,10-14,15-19):")
    for i, w in enumerate(kl['windows']):
        print(f"    Window {i}: {w:.4f}")
    
    kl_warning = ""
    if kl['final_kl'] < 0.01:
        kl_warning = " [WARNING: Policy frozen - LR too low?]"
    elif kl['final_kl'] > 1.0:
        kl_warning = " [WARNING: Policy unstable - LR too high?]"
    print(f"  {kl_warning}")
    
    # C. Scenario coverage
    print("\n" + "=" * 50)
    print("C. SCENARIO COVERAGE")
    print("=" * 50)
    cov = compute_scenario_coverage(episodes)
    print(f"  Unique scenarios: {cov['unique_count']}")
    print(f"  Scenarios: {', '.join(sorted(cov['unique_scenarios'])[:10])}")
    if cov['unique_count'] < 10:
        for sid, cnt in sorted(cov['counts'].items()):
            print(f"    {sid}: {cnt}")
    
    coverage_warning = ""
    if cov['unique_count'] < 3:
        coverage_warning = " [ERROR: <3 unique scenarios - SAMPLING BUG]"
    print(f"  {coverage_warning}")
    
    # D. Reward breakdown
    print("\n" + "=" * 50)
    print("D. REWARD COMPONENT BREAKDOWN")
    print("=" * 50)
    breakdown = compute_reward_breakdown(episodes)
    for comp, stats in breakdown.items():
        print(f"  {comp}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]")
    
    # Check for broken/dominating detectors
    broken = [k for k, v in breakdown.items() if abs(v['mean']) < 0.01]
    dominating = [k for k, v in breakdown.items() if abs(v['mean']) > 1.0]
    if broken:
        print(f"  [WARNING] Always ~0: {broken}")
    if dominating:
        print(f"  [WARNING] Dominating: {dominating}")
    
    # VERDICT
    print("\n" + "=" * 50)
    print("VERDICT")
    print("=" * 50)
    
    # Check conditions for LAUNCH
    return_positive = trend.get('delta', 0) > 0
    kl_healthy = 0.01 <= kl['final_kl'] <= 0.5
    coverage_good = cov['unique_count'] >= 3
    
    issues = []
    if not return_positive:
        issues.append(f"Return delta not positive ({trend.get('delta', 0):.3f})")
    if not kl_healthy:
        if kl['final_kl'] < 0.01:
            issues.append(f"KL too low ({kl['final_kl']:.4f}) - policy frozen")
        elif kl['final_kl'] > 0.5:
            issues.append(f"KL too high ({kl['final_kl']:.4f}) - policy unstable")
    if not coverage_good:
        issues.append(f"Scenario coverage only {cov['unique_count']} - need >=3")
    
    if not issues:
        print("LAUNCH 200-EP")
        print("  - Return delta positive")
        print("  - KL healthy (0.01–0.5 range)")
        print("  - Scenario coverage >= 3")
    else:
        print("TUNE BEFORE 200-EP")
        for issue in issues:
            print(f"  - {issue}")
        
        # Hyperparameter recommendations
        print("\n  Recommended hyperparameter changes:")
        if kl['final_kl'] < 0.01:
            print("    - Increase LR (e.g., 1e-5 -> 2e-5 or 3e-5)")
            print("    - OR decrease kl-coef (e.g., 0.05 -> 0.02)")
        if kl['final_kl'] > 0.5:
            print("    - Decrease LR (e.g., 1e-5 -> 5e-6)")
            print("    - OR increase kl-coef (e.g., 0.05 -> 0.1)")
        if cov['unique_count'] < 3:
            print("    - INVESTIGATE: Check scenario sampler in train_grpo.py")
    
    print()


if __name__ == "__main__":
    main()