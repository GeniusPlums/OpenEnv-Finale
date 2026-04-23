import json
from pathlib import Path
from role_drift_env.models import AgentAction
from role_drift_env.server.environment import RoleDriftEnvironment
from role_drift_env.server.customer_sim import CustomerSimulator
from training.rollout import rollout_episode


def test_rollout_loop():
    """Test that rollout_episode runs end-to-end and produces a trajectory."""
    def policy(obs, state):
        # Simple policy: end call after 3 turns or if customer farewells
        if state.turn_idx >= 3:
            return AgentAction(utterance="Goodbye.", end_call=True)
        if state.customer_farewell_turn is not None and state.disengagement_counter >= 2:
            return AgentAction(utterance="Goodbye, have a great day!", end_call=True)
        return AgentAction(utterance="Let me help you with that.", end_call=False)

    env = RoleDriftEnvironment()
    traj, ret = rollout_episode(
        policy=policy,
        scenario_id="coop_kk_01",
        env=env,
        rollout_idx=0,
    )

    assert len(traj) > 0, "Trajectory should not be empty"
    assert ret != 0 or len(traj) <= 3, "Episode return should be calculable"

    # Verify structure
    for obs, action, reward in traj:
        assert isinstance(action.utterance, str)
        assert isinstance(reward.total, float)
        assert "task" in reward.components

    print(f"Rollout test passed: {len(traj)} turns, return={ret:.3f}")


def test_environment_reset_step():
    """Direct test of env reset and step."""
    env = RoleDriftEnvironment()
    obs, state = env.reset(scenario_id="term_kk_02", rollout_idx=0)
    assert obs.customer_message == state.scenario.opening_message

    sim = CustomerSimulator.from_scenario(state.scenario)
    action = AgentAction(utterance="Hello, how can I help?")
    obs2, reward, done, info = env.step(state, action, sim)

    assert obs2.turn_idx == 1
    assert not done
    assert isinstance(reward.total, float)
    assert len(state.history) == 2  # agent + customer

    print("Environment reset/step test passed")


if __name__ == "__main__":
    test_rollout_loop()
    test_environment_reset_step()
    print("All rollout tests passed!")
