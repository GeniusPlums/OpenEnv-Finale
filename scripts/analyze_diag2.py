#!/usr/bin/env python3
"""Simple analysis for diagnostic run v2."""
import json
from huggingface_hub import hf_hub_download
from statistics import mean, stdev
import numpy as np

# Load from diag2
path = hf_hub_download("GeniusPlums/role-drift-runs-diag2", "episode_log.jsonl")
episodes = []
with open(path, encoding="utf-8") as f:
    for line in f:
        episodes.append(json.loads(line))

print(f"Loaded {len(episodes)} episodes\n")

# A. Return trend
returns = [ep["mean_return"] for ep in episodes]
early_mean = mean(returns[:5])
late_mean = mean(returns[15:])
delta = late_mean - early_mean

coeffs = np.polyfit(list(range(len(returns))), returns, 1)
slope = coeffs[0]

print("=" * 50)
print("A. RETURN TREND (diag2)")
print("=" * 50)
print(f"  Episodes 0-4 mean: {early_mean:.3f}")
print(f"  Episodes 15-19 mean: {late_mean:.3f}")
print(f"  Delta: {delta:.3f}")
print(f"  Slope: {slope:.4f}")

# B. KL
kls = [ep["avg_kl"] for ep in episodes]
init_kl = mean(kls[:5])
final_kl = mean(kls[-5:])

print("\n" + "=" * 50)
print("B. KL TRAJECTORY (diag2)")
print("=" * 50)
print(f"  Initial KL: {init_kl:.4f}")
print(f"  Final KL: {final_kl:.4f}")

# Compare to diag1
print("\n" + "=" * 50)
print("COMPARISON: diag1 vs diag2")
print("=" * 50)
print(f"  diag1: -2.34 -> -5.80 (delta -3.46)")
print(f"  diag2: {early_mean:.2f} -> {late_mean:.2f} (delta {delta:.2f})")
if delta > -3.46:
    print("  ==> IMPROVEMENT: Less negative delta!")
else:
    print("  ==> Still degrading")

# Verdict
print("\n" + "=" * 50)
print("VERDICT")
print("=" * 50)
if delta > -2.0:
    print("IMPROVING: Returns are more stable, less negative trend")
    print("Ready for 200-ep run")
elif delta > -3.46:
    print("SLIGHTLY IMPROVING: Delta improved from -3.46 to {:.2f}".format(delta))
    print("But still negative - investigate further before 200-ep")
else:
    print("STILL DEGRADING: Rewards getting worse")
    print("Investigate before next run")