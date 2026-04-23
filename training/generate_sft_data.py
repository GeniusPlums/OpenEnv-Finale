import json
from pathlib import Path
from role_drift_env.models import State, AgentAction
from role_drift_env.server.environment import RoleDriftEnvironment
from role_drift_env.server.customer_sim import CustomerSimulator


def generate_sft_conversations(
    scenarios_path: str = "data/scenarios/train.jsonl",
    output_path: str = "data/sft/warmstart_conversations.jsonl",
    n_conversations: int = 400,
    agent_policy: str = "frontier",  # "frontier" or "heuristic"
):
    """Generate conversations for SFT warm-start.

    For now, without a frontier API key, we use a simple heuristic policy:
    - Responds with on-topic filler
    - Ends call when customer farewells twice
    """
    env = RoleDriftEnvironment()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load scenario IDs
    scenario_ids = []
    with open(scenarios_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            scenario_ids.append(obj["scenario_id"])

    records = []
    for i in range(n_conversations):
        sid = scenario_ids[i % len(scenario_ids)]
        obs, state = env.reset(scenario_id=sid, rollout_idx=i)
        sim = CustomerSimulator.from_scenario(state.scenario)

        history = []
        done = False
        turn = 0
        while not done and turn < state.scenario.max_turns:
            # Heuristic agent
            if state.customer_farewell_turn is not None and state.disengagement_counter >= 2:
                action = AgentAction(utterance="Goodbye, have a great day!", end_call=True)
            else:
                # Simple on-topic response
                if "workshop" in state.scenario.task_description.lower():
                    action = AgentAction(utterance="Our workshop covers technical and fundamental analysis.", end_call=False)
                elif "application" in state.scenario.task_description.lower():
                    action = AgentAction(utterance="I can help you resume your application. What step did you leave at?", end_call=False)
                else:
                    action = AgentAction(utterance="Let me assist you with that.", end_call=False)

            obs_next, reward, done, info = env.step(state, action, sim)
            history.append({
                "turn_idx": turn,
                "agent_text": action.utterance,
                "customer_text": obs_next.customer_message,
                "reward_components": reward.components,
            })
            turn += 1
            obs = obs_next

        episode_return = sum(h["reward_components"]["task"] for h in history)  # simplified
        terminal = env.check_terminal_success(state)
        records.append({
            "scenario_id": sid,
            "rollout_idx": i,
            "history": history,
            "episode_return": episode_return,
            "terminal_success": terminal,
            "turns": turn,
        })

    # Keep top ~30%
    records.sort(key=lambda x: x["episode_return"], reverse=True)
    keep_count = max(1, int(len(records) * 0.3))
    kept = records[:keep_count]

    with open(output_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    print(f"Generated {len(records)} conversations, kept {keep_count} to {output_path}")
    return kept


if __name__ == "__main__":
    generate_sft_conversations()
