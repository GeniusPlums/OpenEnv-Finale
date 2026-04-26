"""
Reward hacking probes: four trivial full-episode policies vs optional trained checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

print(
    f"[V10] ROLE_DRIFT_PERSONA_OPENAI_BASE_URL={os.environ.get('ROLE_DRIFT_PERSONA_OPENAI_BASE_URL', '')!r}",
    flush=True,
)

from role_drift_env.models import AgentAction, Observation, State
from role_drift_env.server.environment import RoleDriftEnvironment
from training.rollout import rollout_episode

_FAREWELL_RE = re.compile(
    r"(?i)(goodbye|bye\b|thanks|thank you|have a good day|see you|take care)",
)


def _farewell_in_history(state: State) -> bool:
    for turn in state.history:
        if turn.get("role") != "customer":
            continue
        text = turn.get("text") or ""
        if _FAREWELL_RE.search(text):
            return True
    return False


def policy_always_empty(_obs: Observation, _state: State) -> AgentAction:
    return AgentAction(utterance="")


def policy_always_rephrase(_obs: Observation, _state: State) -> AgentAction:
    return AgentAction(
        utterance="Let me rephrase that in simpler terms: I understand your message and I am here to help you move forward."
    )


def policy_always_summary(_obs: Observation, _state: State) -> AgentAction:
    return AgentAction(
        utterance="In short: the key takeaway is to stay focused on the next step the college asked for and avoid unnecessary tangents."
    )


def policy_mute_after_farewell(obs: Observation, state: State) -> AgentAction:
    if _FAREWELL_RE.search((obs.customer_message or "").strip()) or _farewell_in_history(
        state
    ):
        return AgentAction(utterance="")
    return AgentAction(
        utterance="I can help you with the application. What is the one thing you need help with first?"
    )


POLICIES = {
    "always_empty": policy_always_empty,
    "always_rephrase": policy_always_rephrase,
    "always_summary": policy_always_summary,
    "mute_after_farewell": policy_mute_after_farewell,
}


def _run_one_episode(
    env: RoleDriftEnvironment,
    policy,
    scenario_id: str,
    seed: int,
    max_turns: int,
) -> float:
    traj, ret, _st = rollout_episode(
        policy=policy,
        scenario_id=scenario_id,
        env=env,
        rollout_idx=seed,
        max_turns_override=max_turns,
        return_state=True,
    )
    return float(ret)


def run_policy_over_file(
    policy_name: str,
    scenario_path: Path,
    seeds: list[int],
    max_turns: int,
) -> tuple[list[float], float, float, float]:
    policy = POLICIES[policy_name]
    env = RoleDriftEnvironment()
    rows = _load_scenarios(scenario_path)
    returns: list[float] = []
    for s in rows:
        sid = s["scenario_id"]
        for seed in seeds:
            returns.append(
                _run_one_episode(env, policy, sid, seed, max_turns),
            )
    m = float(np.mean(returns)) if returns else 0.0
    if len(returns) < 2:
        return returns, m, m, m
    rng = np.random.default_rng(42)
    boot = 2000
    means = np.empty(boot, dtype=np.float64)
    arr = np.asarray(returns, dtype=np.float64)
    n = len(arr)
    for i in range(boot):
        means[i] = float(np.mean(rng.choice(arr, size=n, replace=True)))
    return (
        returns,
        m,
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def _load_scenarios(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _trained_mean_from_checkpoint(
    checkpoint: str, scenario_path: Path, seeds: list[int]
) -> float | None:
    try:
        from training.eval_baseline import evaluate_baseline
    except Exception as e:
        print(f"[probes] trained eval import failed: {e}", file=sys.stderr)
        return None
    out = Path("/tmp/probe_trained_eval.json")
    if sys.platform == "win32":
        out = Path("data/eval_results/_tmp_probe_trained.json")
    payload = evaluate_baseline(
        model_path=checkpoint,
        scenario_file=str(scenario_path),
        num_seeds=max(seeds) + 1 if seeds else 3,
        output_path=str(out),
    )
    return float(payload["summary"]["mean_return"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Trivial full-episode policies (reward-hacking baselines).",
    )
    ap.add_argument(
        "--scenarios-jsonl",
        "--scenario-file",
        default="data/scenarios/eval.jsonl",
        dest="scenarios_jsonl",
        help="Scenarios (eval.jsonl, never train.jsonl).",
    )
    ap.add_argument(
        "--output-json",
        "--output",
        default="data/eval_results/reward_hacking_probes.json",
        dest="output_json",
    )
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="Optional trained model path or repo id to log reference mean (full eval).",
    )
    ap.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma-separated rollout seeds (same as V10 eval).",
    )
    ap.add_argument(
        "--max-turns", type=int, default=6, help="Max turns (match V9 training / run_eval)."
    )
    args = ap.parse_args()
    scenario_path = Path(args.scenarios_jsonl)
    if "train" in str(scenario_path).lower():
        print("FATAL: do not use training scenarios for probes.", file=sys.stderr)
        sys.exit(1)
    if not scenario_path.is_file():
        print(f"Not found: {scenario_path}", file=sys.stderr)
        sys.exit(1)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    out: dict = {
        "scenarios_file": str(scenario_path),
        "n_scenarios": len(_load_scenarios(scenario_path)),
        "n_seeds": len(seeds),
        "max_turns": args.max_turns,
        "policies": {},
    }

    for name in POLICIES:
        rets, m, lo, hi = run_policy_over_file(
            name, scenario_path, seeds, args.max_turns
        )
        out["policies"][name] = {
            "mean_total_reward": round(m, 4),
            "ci_lower_95": round(lo, 4),
            "ci_upper_95": round(hi, 4),
            "n_values": len(rets),
        }
        print(f"[probes] {name}: mean={m:.4f} ci=[{lo:.4f}, {hi:.4f}]")

    out["trained_reference_mean_return"] = None
    if args.checkpoint:
        tm = _trained_mean_from_checkpoint(
            args.checkpoint, scenario_path, seeds
        )
        if tm is not None:
            out["trained_reference_mean_return"] = round(tm, 4)
            print(f"[probes] trained reference (evaluate_baseline): {tm:.4f}")

    outp = Path(args.output_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved {outp}")


if __name__ == "__main__":
    main()
