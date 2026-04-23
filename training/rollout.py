import json
from pathlib import Path
from typing import Callable, List, Tuple, Optional
from role_drift_env.models import AgentAction, Observation, State, TurnReward
from role_drift_env.server.environment import RoleDriftEnvironment
from role_drift_env.server.customer_sim import CustomerSimulator


def rollout_episode(
    policy: Callable[[Observation, State], AgentAction],
    scenario_id: str,
    env: RoleDriftEnvironment = None,
    rollout_idx: int = 0,
    transcript_dir: Optional[str] = None,
) -> Tuple[List[Tuple[Observation, AgentAction, TurnReward]], float]:
    """Run one full episode with the given policy.

    Args:
        transcript_dir: if provided, saves a human-readable transcript JSON

    Returns:
        trajectory: list of (observation, action, reward) tuples
        episode_return: sum of all turn rewards + terminal success
    """
    if env is None:
        env = RoleDriftEnvironment()
    obs, state = env.reset(scenario_id=scenario_id, rollout_idx=rollout_idx)
    sim = CustomerSimulator.from_scenario(state.scenario)

    trajectory = []
    transcript = {
        "scenario_id": scenario_id,
        "rollout_idx": rollout_idx,
        "turns": [],
    }
    episode_return = 0.0

    done = False
    while not done:
        action = policy(obs, state)
        obs_next, reward, done, info = env.step(state, action, sim)
        trajectory.append((obs, action, reward))
        episode_return += reward.total

        # Build transcript turn
        transcript["turns"].append({
            "turn_idx": state.turn_idx,
            "customer_message": obs.customer_message,
            "agent_utterance": action.utterance,
            "agent_end_call": action.end_call,
            "reward_total": reward.total,
            "reward_components": reward.components,
        })

        obs = obs_next
        if state.turn_idx >= state.scenario.max_turns:
            done = True

    # Terminal success
    terminal = env.check_terminal_success(state)
    episode_return += terminal
    transcript["episode_return"] = episode_return
    transcript["terminal_success"] = terminal
    transcript["num_turns"] = len(transcript["turns"])

    if transcript_dir:
        out_dir = Path(transcript_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{scenario_id}_r{rollout_idx}.json"
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)

    return trajectory, episode_return
