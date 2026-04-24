#!/usr/bin/env python3
"""
Deep diagnostic: Run a single episode with verbose reward logging.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from role_drift_env.server.environment import RoleDriftEnvironment
from role_drift_env.server.customer_sim import CustomerSimulator
from role_drift_env.models import AgentAction

def run_episode_verbose(scenario_id):
    """Run one episode with full reward debugging."""
    env = RoleDriftEnvironment()
    obs, state = env.reset(scenario_id=scenario_id, rollout_idx=0)
    sim = CustomerSimulator.from_scenario(state.scenario)
    
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_id}")
    print(f"Task: {state.scenario.task_description}")
    print(f"Persona: {state.scenario.persona_id}")
    print(f"Drift types: {state.scenario.drift_types}")
    print(f"Explicit rules: {len(state.scenario.explicit_rules)}")
    print(f"{'='*60}\n")
    
    total_reward = 0.0
    
    # Simple fixed policy: echo a generic response
    def simple_policy(obs, state):
        # Just use a simple fixed response based on turn
        responses = [
            "I'd be happy to help you with that. Could you tell me more?",
            "Sure, let me explain our workshop details.",
            "The workshop covers technical analysis and fundamental trading.",
            "Our fee is rupees four nine nine.",
            "Thank you for your interest!",
        ]
        turn = state.turn_idx if state.turn_idx < len(responses) else -1
        text = responses[turn] if turn >= 0 else "Thank you, goodbye!"
        end_call = "goodbye" in text.lower() or "thank you" in text.lower() and state.turn_idx > 3
        return AgentAction(utterance=text, end_call=end_call)
    
    done = False
    turn = 0
    while not done and turn < 15:
        action = simple_policy(obs, state)
        
        # Debug: print what the policy would generate
        print(f"Turn {turn}:")
        print(f"  Customer: {obs.customer_message[:80]}")
        print(f"  Agent: {action.utterance[:80]}")
        
        # Step the environment
        obs_next, reward, done, info = env.step(state, action, sim)
        
        print(f"  Reward total: {reward.total:.3f}")
        print(f"  Reward components: {reward.components}")
        print(f"  State: turn_idx={state.turn_idx}, farewell={state.customer_farewell_turn}, diseng={state.disengagement_counter}")
        print()
        
        total_reward += reward.total
        obs = obs_next
        turn += 1
        
        if state.turn_idx >= state.scenario.max_turns:
            break
    
    # Terminal success
    terminal = env.check_terminal_success(state)
    print(f"Terminal success: {terminal}")
    total_reward += terminal
    
    print(f"\nFinal episode return: {total_reward:.3f}")
    return total_reward


# Run on a few different scenario types
for sid in ["term_kk_05", "goal_kk_01", "instr_kk_01", "coop_kk_01"]:
    try:
        run_episode_verbose(sid)
    except Exception as e:
        print(f"Error on {sid}: {e}")
        import traceback
        traceback.print_exc()