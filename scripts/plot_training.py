"""
Plot mean return and per-component means from episode_log.jsonl (GRPO training).
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("episode_log", type=Path, help="Path to episode_log.jsonl")
    p.add_argument("--out", type=Path, default=Path("plots/run_final"), help="Output directory")
    args = p.parse_args()

    episodes = []
    mean_returns = []
    comp_keys: set[str] = set()
    rows = []
    with open(args.episode_log, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            rows.append(o)
            episodes.append(o["episode"])
            mean_returns.append(o["mean_return"])
            for k, v in (o.get("component_means") or {}).items():
                if isinstance(v, (int, float)):
                    comp_keys.add(k)

    args.out.mkdir(parents=True, exist_ok=True)

    # Overall reward
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, mean_returns, label="mean_return", alpha=0.6)
    w = 20
    if len(mean_returns) >= w:
        roll = np.convolve(mean_returns, np.ones(w) / w, mode="valid")
        plt.plot(episodes[w - 1 :], roll, label=f"rolling_{w}", linewidth=2)
    plt.xlabel("episode")
    plt.ylabel("mean return (group)")
    plt.title("GRPO — mean return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out / "reward_curve.png", dpi=120)
    plt.close()
    print(f"Wrote {args.out / 'reward_curve.png'}")

    # Per-component
    for key in sorted(comp_keys):
        ys = [float((r.get("component_means") or {}).get(key, np.nan)) for r in rows]
        if all(np.isnan(ys)):
            continue
        plt.figure(figsize=(10, 4))
        plt.plot(episodes, ys, label=key)
        if len(ys) >= 20:
            arr = np.array(ys, dtype=float)
            roll = np.convolve(arr, np.ones(20) / 20, mode="valid")
            plt.plot(episodes[19:], roll, label="rolling_20", linewidth=2)
        plt.xlabel("episode")
        plt.ylabel("mean (group)")
        plt.title(f"Component: {key}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe = key.replace(os.sep, "_").replace(":", "_")
        plt.savefig(args.out / f"component_{safe}.png", dpi=120)
        plt.close()
        print(f"Wrote {args.out / f'component_{safe}.png'}")

    # Stacked 2x2 style single figure if we have 4 common keys
    order = [k for k in ["term", "goal", "instr", "lang", "task"] if k in comp_keys]
    if order:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
        axes = axes.ravel()
        for ax, k in zip(axes, order[:4]):
            ys = [float((r.get("component_means") or {}).get(k, 0.0)) for r in rows]
            ax.plot(episodes, ys, label=k, color="#2a9d8f")
            ax.set_title(k)
            ax.grid(True, alpha=0.3)
        for j in range(len(order[:4]), 4):
            axes[j].set_visible(False)
        plt.suptitle("Per-component (group-mean) training curves")
        plt.tight_layout()
        plt.savefig(args.out / "reward_curve_by_component.png", dpi=120)
        plt.close()
        print(f"Wrote {args.out / 'reward_curve_by_component.png'}")


if __name__ == "__main__":
    main()
