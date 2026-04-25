"""One full-episode rollout with a local checkpoint; writes JSON for closeout sanity."""
import argparse
import json
import sys
from pathlib import Path

# Run from repo root
if __name__ == "__main__":
    from training.eval_baseline import _make_local_model_policy
    from training.rollout import rollout_episode

    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--scenario-file", default="data/scenarios/eval.jsonl")
    p.add_argument("--output", default="/tmp/sanity_rollout.json")
    args = p.parse_args()

    with open(args.scenario_file, encoding="utf-8") as f:
        first = json.loads(f.readline())
    sid = first["scenario_id"]

    pol = _make_local_model_policy(args.model_path)
    from role_drift_env.server.environment import RoleDriftEnvironment

    env = RoleDriftEnvironment()
    traj, ret = rollout_episode(pol, sid, env=env, rollout_idx=0)
    # turns: list of (obs, action, reward) — need action text from first agent step
    first_agent = ""
    for obs, act, _rew in traj:
        if act and act.utterance is not None:
            first_agent = act.utterance
            break
    out = {
        "scenario_id": sid,
        "episode_return": ret,
        "first_agent_utterance": first_agent,
        "n_turns": len(traj),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out)[:2000])
