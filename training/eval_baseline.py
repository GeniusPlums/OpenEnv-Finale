import json
from pathlib import Path
from role_drift_env.models import AgentAction
from role_drift_env.server.environment import RoleDriftEnvironment
from role_drift_env.server.customer_sim import CustomerSimulator
from rollout import rollout_episode


def evaluate_baseline(
    policy_type: str,
    scenario_file: str = "data/scenarios/eval.jsonl",
    num_seeds: int = 3,
    output_path: str = None,
):
    """Evaluate a baseline policy."""
    if output_path is None:
        output_path = f"eval_{policy_type}.jsonl"

    env = RoleDriftEnvironment()

    def frontier_like_policy(obs, state):
        # Simulated frontier model behavior: generally good but drifts occasionally
        if state.turn_idx >= 5:
            return AgentAction(utterance="Thank you for your time. Goodbye!", end_call=True)
        if "workshop" in state.scenario.task_description.lower():
            return AgentAction(utterance="Our workshop covers technical and fundamental analysis.", end_call=False)
        return AgentAction(utterance="I can help you with that. What do you need?", end_call=False)

    def deployable_like_policy(obs, state):
        # Simulated deployable model: cheaper, faster, more drift
        if state.turn_idx >= 8:
            return AgentAction(utterance="You are welcome. Have a nice day!", end_call=False)  # forgets to end
        if "workshop" in state.scenario.task_description.lower():
            return AgentAction(utterance="The fee is 499 rupees. 1. Technical analysis 2. Fundamental analysis", end_call=False)
        return AgentAction(utterance="Let me help you with your application.", end_call=False)

    def sft_policy(obs, state):
        # SFT checkpoint: better than deployable, not fully trained
        if state.customer_farewell_turn is not None and state.disengagement_counter >= 2:
            return AgentAction(utterance="Goodbye, have a great day!", end_call=True)
        if "workshop" in state.scenario.task_description.lower():
            return AgentAction(utterance="Our workshop covers technical and fundamental analysis.", end_call=False)
        return AgentAction(utterance="I can help you resume your application. Where did you leave off?", end_call=False)

    policies = {
        "frontier": frontier_like_policy,
        "deployable": deployable_like_policy,
        "sft": sft_policy,
    }

    policy = policies.get(policy_type, sft_policy)

    scenario_ids = []
    with open(scenario_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            scenario_ids.append(obj["scenario_id"])

    results = []
    for sid in scenario_ids:
        for seed in range(num_seeds):
            traj, ret = rollout_episode(
                policy=policy,
                scenario_id=sid,
                env=env,
                rollout_idx=seed,
            )
            results.append({
                "scenario_id": sid,
                "seed": seed,
                "episode_return": ret,
                "turns": len(traj),
            })
            print(f"{sid} seed={seed} return={ret:.3f} turns={len(traj)}")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    returns = [r["episode_return"] for r in results]
    print(f"[{policy_type}] Mean return: {sum(returns)/len(returns):.3f}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["frontier", "deployable", "sft"], required=True)
    parser.add_argument("--scenario-file", default="data/scenarios/eval.jsonl")
    parser.add_argument("--num-seeds", type=int, default=3)
    args = parser.parse_args()
    evaluate_baseline(args.policy, args.scenario_file, args.num_seeds)
