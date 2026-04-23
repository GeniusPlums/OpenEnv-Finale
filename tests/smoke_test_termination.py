import pytest
from role_drift_env.models import AgentAction
from role_drift_env.server.environment import RoleDriftEnvironment
from role_drift_env.server.customer_sim import CustomerSimulator
from training.rollout import rollout_episode


def looping_policy(obs, state):
    """A known-bad agent that always says 'you are welcome' and never ends the call."""
    return AgentAction(utterance="You are welcome.", end_call=False)


def test_smoke_test_termination():
    env = RoleDriftEnvironment()
    trajectory, episode_return = rollout_episode(
        policy=looping_policy,
        scenario_id="term_kk_02",
        env=env,
        rollout_idx=0,
    )
    # Assert episode return is strongly negative due to termination penalty
    print(f"Episode return: {episode_return}")
    assert episode_return < -2.0, f"Expected strongly negative return, got {episode_return}"
    # Assert we ran multiple turns
    assert len(trajectory) >= 5
    # Assert at least some turns had negative termination component
    term_penalties = [r.components.get("term", 0) for _, _, r in trajectory]
    assert any(t < 0 for t in term_penalties), "Expected some termination penalties"


if __name__ == "__main__":
    test_smoke_test_termination()
    print("Smoke test passed!")
