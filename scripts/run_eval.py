"""
V10 eval harness: run trained or baseline policy on held-out scenarios, upload partial JSON to Hub.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# FM-1: first log line in process (grep-friendly)
print(
    f"[V10] ROLE_DRIFT_PERSONA_OPENAI_BASE_URL={os.environ.get('ROLE_DRIFT_PERSONA_OPENAI_BASE_URL', '')!r}",
    flush=True,
)

from role_drift_env.server.environment import RoleDriftEnvironment
from training.eval_baseline import _make_local_model_policy
from training.rollout import rollout_episode

_DEFAULT_EVAL = "data/scenarios/eval.jsonl"
_DEFAULT_TRANSFER = "data/scenarios/transfer_dearconnect.jsonl"
_TRAIN_PATH = "data/scenarios/train.jsonl"


def _load_jsonl_scenarios(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _train_scenario_ids() -> set[str]:
    p = Path(_TRAIN_PATH)
    if not p.is_file():
        return set()
    ids: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        ids.add(json.loads(line)["scenario_id"])
    return ids


def _primary_drift(s: dict[str, Any]) -> str:
    types = s.get("drift_types") or []
    for t in types:
        if t != "cooperative":
            return str(t)
    return "cooperative"


def _domain_label(prompt_id: str) -> str:
    return {
        "kundan_kishore": "kundan_kishore",
        "masters_union": "masters_union",
        "dearconnect": "dearconnect",
    }.get(prompt_id, prompt_id)


def _parse_seeds(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _bootstrap_mean_ci(
    values: list[float], n_boot: int = 2000, seed: int = 42, alpha: float = 0.05
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = float(np.mean(arr))
    if n < 2:
        return m, m, m
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = float(np.mean(sample))
    return m, float(np.percentile(means, alpha / 2 * 100)), float(np.percentile(means, (1 - alpha / 2) * 100))


def _aggregate(
    results: list[dict[str, Any]], scenarios_by_id: dict[str, dict]
) -> dict[str, Any]:
    totals = [r["total_reward"] for r in results]
    m, lo, hi = _bootstrap_mean_ci(totals)
    by_dt: dict[str, list[float]] = defaultdict(list)
    for r in results:
        sid = r["scenario_id"]
        meta = scenarios_by_id.get(sid, {})
        dt = r.get("drift_type") or _primary_drift(meta)
        by_dt[dt].append(r["total_reward"])
    by_drift: dict[str, Any] = {}
    for dt, vals in by_dt.items():
        mean_dt, lo_dt, hi_dt = _bootstrap_mean_ci(vals)
        by_drift[dt] = {
            "mean": round(mean_dt, 4),
            "n": len(vals),
            "ci_lower": round(lo_dt, 4),
            "ci_upper": round(hi_dt, 4),
        }
    return {
        "mean_total_reward": round(m, 4),
        "ci_lower_95": round(lo, 4),
        "ci_upper_95": round(hi, 4),
        "by_drift_type": by_drift,
    }


def _row_from_rollout(
    scenario_meta: dict[str, Any],
    seed: int,
    traj: list,
    episode_return: float,
    state,
    env: RoleDriftEnvironment,
) -> dict[str, Any]:
    per_det: dict[str, float] = {"term": 0.0, "goal": 0.0, "instr": 0.0, "lang": 0.0}
    task_sum = 0.0
    per_turn: list[dict[str, Any]] = []
    for _obs, _action, rw in traj:
        for k, v in rw.components.items():
            if k == "task":
                task_sum += v
            elif k in per_det:
                per_det[k] += v
        per_turn.append(
            {
                "reward_total": round(rw.total, 4),
                "reward_components": {k: round(v, 4) for k, v in rw.components.items()},
            }
        )
    ts = float(env.check_terminal_success(state)) > 0.0
    agent_lines: list[str] = []
    customer_lines: list[str] = []
    for obs, action, _rw in traj:
        customer_lines.append(obs.customer_message or "")
        agent_lines.append(action.utterance)
    return {
        "scenario_id": scenario_meta["scenario_id"],
        "drift_type": _primary_drift(scenario_meta),
        "domain": _domain_label(scenario_meta.get("prompt_id", "")),
        "seed": seed,
        "total_reward": round(float(episode_return), 4),
        "per_detector_total": {k: round(v, 4) for k, v in per_det.items()},
        "task_bonus": round(task_sum, 4),
        "turn_count": len(traj),
        "terminal_success": bool(ts),
        "per_turn": per_turn,
        "agent_transcript": agent_lines,
        "customer_transcript": customer_lines,
    }


def _upload_eval_results(repo: str, local_dir: Path) -> None:
    """Upload files under data/eval_results/ to a Hub dataset repo (FM-2)."""
    if not repo:
        return
    try:
        subprocess.run(
            [
                "hf",
                "upload",
                repo,
                str(local_dir),
                ".",
                "--repo-type",
                "dataset",
            ],
            check=False,
            timeout=600,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        print(f"[run_eval] WARN: upload failed: {e}", file=sys.stderr, flush=True)


def _run_subcommand(args: argparse.Namespace) -> int:
    scenario_path = Path(args.scenarios_jsonl)
    if "train" in str(scenario_path).lower():
        print(
            f"FATAL: eval path must not reference training data (got {scenario_path})",
            file=sys.stderr,
        )
        return 1

    scenarios = _load_jsonl_scenarios(scenario_path)
    train_ids = _train_scenario_ids()
    eval_ids = {s["scenario_id"] for s in scenarios}
    if train_ids and eval_ids and not eval_ids.isdisjoint(train_ids):
        inter = eval_ids & train_ids
        print(f"FATAL: eval scenarios overlap with train: {sorted(inter)[:5]}", file=sys.stderr)
        return 1
    if scenario_path.name == "eval.jsonl" and len(eval_ids) > 10:
        print(
            f"FATAL: expected at most 10 in-domain eval scenarios, got {len(eval_ids)}",
            file=sys.stderr,
        )
        return 1

    print(f"[run_eval] scenarios_file={args.scenarios_jsonl}", flush=True)
    print(f"[run_eval] num_scenarios={len(scenarios)}", flush=True)
    print(
        f"[run_eval] first_scenario_id={scenarios[0]['scenario_id'] if scenarios else 'NONE'}",
        flush=True,
    )
    print(
        f"[run_eval] policy={args.policy_checkpoint} label={args.label}",
        flush=True,
    )

    scenarios_by_id = {s["scenario_id"]: s for s in scenarios}
    seeds = _parse_seeds(args.seeds)
    env = RoleDriftEnvironment()
    policy = _make_local_model_policy(args.policy_checkpoint)

    results: list[dict[str, Any]] = []
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def build_payload() -> dict[str, Any]:
        return {
            "label": args.label,
            "checkpoint": args.policy_checkpoint,
            "scenarios_file": args.scenarios_jsonl,
            "n_scenarios": len(scenarios),
            "n_seeds": len(seeds),
            "results": results,
            "aggregate": _aggregate(results, scenarios_by_id),
        }

    for s in scenarios:
        sid = s["scenario_id"]
        for seed in seeds:
            traj, ret, st = rollout_episode(
                policy=policy,
                scenario_id=sid,
                env=env,
                rollout_idx=seed,
                max_turns_override=args.max_turns,
                return_state=True,
            )
            row = _row_from_rollout(s, seed, traj, ret, st, env)
            results.append(row)
            print(
                f"[run_eval] {args.label} {sid} seed={seed} total_reward={row['total_reward']}",
                flush=True,
            )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(build_payload(), f, indent=2, ensure_ascii=False)
        if args.upload_on_each_scenario:
            _upload_eval_results(args.upload_on_each_scenario, out_path.parent)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(build_payload(), f, indent=2, ensure_ascii=False)
    if args.upload_on_each_scenario:
        _upload_eval_results(args.upload_on_each_scenario, out_path.parent)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="V10 policy evaluation (GRPO / baseline)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_run(name: str, default_path: str) -> None:
        sp = sub.add_parser(
            name,
            help=f"Mode {name} (default scenarios: {default_path})",
        )
        sp.add_argument(
            "--scenarios-jsonl",
            default=default_path,
            help="Held-out JSONL (must not be train.jsonl)",
        )
        sp.add_argument("--policy-checkpoint", required=True)
        sp.add_argument("--seeds", default="0,1,2,3,4")
        sp.add_argument("--label", required=True)
        sp.add_argument("--output-json", required=True)
        sp.add_argument(
            "--upload-on-each-scenario",
            default="",
            help="Dataset repo id (e.g. org/name) to push after each scenario",
        )
        sp.add_argument(
            "--max-turns",
            type=int,
            default=6,
            help="Override scenario max turns (V9 training used 6)",
        )
        sp.set_defaults(_default_path=default_path)

    add_run("in_domain", _DEFAULT_EVAL)
    add_run("transfer", _DEFAULT_TRANSFER)
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if hasattr(args, "upload_on_each_scenario") and not args.upload_on_each_scenario:
        args.upload_on_each_scenario = None
    if args.cmd in ("in_domain", "transfer"):
        if not Path(args.scenarios_jsonl).is_file():
            print(f"FATAL: missing {args.scenarios_jsonl}", file=sys.stderr)
            sys.exit(1)
        if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
            if args.upload_on_each_scenario:
                print("FATAL: HF_TOKEN / HUGGINGFACE_HUB_TOKEN required for --upload", file=sys.stderr)
                sys.exit(1)
    rc = _run_subcommand(args)
    # FM-3: after policy, drop references (one model per process; process exits)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
