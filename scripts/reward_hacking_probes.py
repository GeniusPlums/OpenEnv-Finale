"""
Reward hacking probes: trivial one-step policies vs optional full-episode trained policy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from role_drift_env.models import AgentAction
from role_drift_env.server.environment import RoleDriftEnvironment
from role_drift_env.server.rewards import RewardComposer


def run_probe(env, scenario_id: str, agent_mode: str, num_seeds: int = 3) -> float:
    results = []
    for seed in range(num_seeds):
        try:
            obs, state = env.reset(scenario_id, seed)
        except Exception:
            continue

        if agent_mode == "silent":
            action = AgentAction(utterance="")
        elif agent_mode == "minimal":
            action = AgentAction(utterance="Okay.")
        elif agent_mode == "verbose":
            action = AgentAction(
                utterance="Thank you for sharing that information. I understand your interest and will consider the appropriate steps to move forward effectively."
            )
        else:
            continue

        composer = RewardComposer()
        reward = composer.score(state, action)
        results.append(reward.total)

    return float(np.mean(results)) if results else 0.0


def run_trivial_probes(scenario_file: Path, num_seeds: int) -> dict[str, float]:
    env = RoleDriftEnvironment()
    with open(scenario_file, encoding="utf-8") as f:
        scenarios = [json.loads(line)["scenario_id"] for line in f][:10]
    results: dict[str, float] = {}
    for mode in ("silent", "minimal", "verbose"):
        total = 0.0
        count = 0
        for sid in scenarios:
            r = run_probe(env, sid, mode, num_seeds=num_seeds)
            if r != 0:
                total += r
                count += 1
        results[mode] = round(total / max(count, 1), 3)
    return results


def run_trained_probe(
    checkpoint: str, scenario_file: Path, num_seeds: int, temp_out: Path
) -> float:
    from training.eval_baseline import evaluate_baseline

    temp_out.parent.mkdir(parents=True, exist_ok=True)
    payload = evaluate_baseline(
        model_path=checkpoint,
        scenario_file=str(scenario_file),
        num_seeds=num_seeds,
        output_path=str(temp_out),
    )
    return float(payload["summary"]["mean_return"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="Trained model dir (optional)")
    ap.add_argument(
        "--scenario-file",
        default="data/scenarios/eval.jsonl",
        help="Scenarios for probes and trained eval",
    )
    ap.add_argument(
        "--output",
        default="data/eval_results/reward_hacking_probe_complete.json",
        help="Output JSON path",
    )
    ap.add_argument("--num-seeds", type=int, default=2, help="Seeds per scenario (trivial + trained)")
    ap.add_argument("--plot", action="store_true", help="Save plots/reward_hacking_probe.png")
    args = ap.parse_args()

    scenario_file = Path(args.scenario_file)
    if not scenario_file.is_file():
        print(f"Not found: {scenario_file}", file=sys.stderr)
        sys.exit(1)

    print("Running trivial probes (one-step)...")
    trivial = run_trivial_probes(scenario_file, args.num_seeds)

    out_obj: dict = {
        "silent": {"aggregate_mean": trivial["silent"]},
        "minimal": {"aggregate_mean": trivial["minimal"]},
        "verbose": {"aggregate_mean": trivial["verbose"]},
        "trained": None,
    }

    if args.checkpoint:
        print("Running trained policy (full episodes)...")
        tmp = Path("/tmp/reward_hacking_trained_eval.json")
        if sys.platform == "win32":
            tmp = Path(args.output).parent / "_tmp_trained_probe_eval.json"
        mean = run_trained_probe(
            args.checkpoint, scenario_file, args.num_seeds, tmp
        )
        out_obj["trained"] = {"aggregate_mean": round(mean, 4)}
        try:
            tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
    else:
        out_obj["trained"] = {"aggregate_mean": None}

    # Backward compatibility
    out_obj["probe_results"] = {
        k: trivial[k] for k in ("silent", "minimal", "verbose")
    }

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)
    print(f"Saved {outp}")

    if args.plot or args.checkpoint:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        modes = ["silent", "minimal", "verbose", "trained"]
        vals = [
            (out_obj[m] or {}).get("aggregate_mean")
            if isinstance(out_obj.get(m), dict)
            else None
            for m in modes
        ]
        vals_plot = [float(v) if v is not None else 0.0 for v in vals]
        labels = ["Silent", "Minimal", "Verbose", "Trained"]
        colors = ["#e76f51", "#e76f51", "#e76f51", "#2a9d8f"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, vals_plot, color=colors)
        ax.set_ylabel("Mean aggregate reward (see docstring for scale mix)")
        ax.set_title("Reward-hacking probes")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)
        Path("plots").mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig("plots/reward_hacking_probe.png", dpi=120)
        plt.close()
        print("Saved plots/reward_hacking_probe.png")


if __name__ == "__main__":
    main()
