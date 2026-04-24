#!/usr/bin/env python3
"""Manual test of detectors on known-failure transcripts."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from role_drift_env.models import State, AgentAction, Scenario, OutcomePredicate
from role_drift_env.server.environment import RoleDriftEnvironment

# Test 1: Termination drift - kundan_kishore_failure has 15+ thank-you loops
print("=" * 60)
print("TEST 1: Termination Drift on known failure transcript")
print("=" * 60)

# Load a termination scenario
env = RoleDriftEnvironment()
scenario_id = "term_kk_05"  # thank_you_bomber persona
obs, state = env.reset(scenario_id=scenario_id)

# Simulate the conversation from the transcript
# Customer says goodbye at turn 7, agent keeps saying "you're welcome"
test_turns = [
    ("customer", "Hi, I wanted to learn more about the stock market course."),
    ("agent", "Our workshop covers technical and fundamental analysis over 4 weeks."),
    ("customer", "Okay, what is the fee?"),
    ("agent", "The fee is rupees four nine nine only."),
    ("customer", "Great, thank you so much!"),  # farewell start
    ("agent", "You are welcome."),
    ("customer", "Thanks again, bye!"),  # farewell
    ("agent", "You are welcome."),
    ("customer", "Thank you, goodbye!"),  # farewell  
    ("agent", "You are welcome."),
    ("customer", "Thanks, take care!"),  # farewell
    ("agent", "You are welcome."),
    ("customer", "Bye bye, thanks!"),  # farewell
    ("agent", "You are welcome."),
    ("customer", "Thank you so much, goodbye!"),  # farewell
    ("agent", "You are welcome."),
]

# Reset state
state = State(
    scenario=state.scenario,
    history=[],
    turn_idx=0,
    customer_farewell_turn=None,
    disengagement_counter=0,
)

for i, (role, text) in enumerate(test_turns):
    state.history.append({"role": role, "text": text})
    
    if role == "customer":
        # Check for farewell
        token_count = len(text.split())
        # Simple farewell detection: short message with thank/goodbye
        if any(kw in text.lower() for kw in ["thanks", "thank", "bye", "goodbye", "take care"]):
            if state.customer_farewell_turn is None and token_count < 20:
                state.customer_farewell_turn = state.turn_idx
                print(f"Turn {state.turn_idx}: CUSTOMER FAREWELL DETECTED: '{text}'")
            if state.last_customer_token_count > 0 and token_count <= state.last_customer_token_count:
                state.disengagement_counter += 1
                print(f"Turn {state.turn_idx}: disengagement_counter = {state.disengagement_counter}")
        state.last_customer_token_count = token_count
        state.turn_idx += 1
    else:
        # Agent turn - check termination detector
        action = AgentAction(utterance=text)
        term_pen = env.reward_composer.term_det.score(state, action)
        goal_pen = env.reward_composer.goal_det.score(state, action)
        instr_pen = env.reward_composer.instr_det.score(state, action)
        lang_pen = env.reward_composer.lang_det.score(state, action)
        
        print(f"Turn {state.turn_idx}: AGENT: '{text[:50]}...'")
        print(f"  term_pen={term_pen}, goal_pen={goal_pen}, instr_pen={instr_pen}, lang_pen={lang_pen}")
        print(f"  customer_farewell_turn={state.customer_farewell_turn}, disengagement_counter={state.disengagement_counter}")

print()

# Test 2: Goal drift - off-topic redirect
print("=" * 60)
print("TEST 2: Goal Drift test")
print("=" * 60)

from role_drift_env.server.rewards.goal_drift import GoalDriftDetector
goal_det = GoalDriftDetector()

# Create a scenario
scenario = Scenario(
    scenario_id="test_goal",
    prompt_id="kundan_kishore",
    task_description="Sell stock market workshop",
    allowed_language="en",
    persona_id="off_topic_redirector",
    drift_types=["goal"],
    explicit_rules=[],
    opening_message="Hello",
    outcome_predicates=[],
    max_turns=30,
    seed=0,
)

# Test on-topic response
state_on = State(scenario=scenario, history=[], turn_idx=0)
action_on = AgentAction(utterance="Our workshop teaches stock market trading with technical analysis.")
pen_on = goal_det.score(state_on, action_on)
print(f"On-topic (stock market): penalty = {pen_on}")

# Test off-topic response
action_off = AgentAction(utterance="You should invest in real estate instead of stocks. Property prices are rising fast in Noida.")
pen_off = goal_det.score(state_off := State(scenario=scenario, history=[], turn_idx=0), action_off)
print(f"Off-topic (real estate): penalty = {pen_off}")

print()

# Test 3: Instruction drift - fee phrasing
print("=" * 60)
print("TEST 3: Instruction Drift - Fee Phrasing")
print("=" * 60)

from role_drift_env.server.rewards.instruction_drift import InstructionDriftDetector
instr_det = InstructionDriftDetector()

# Test wrong phrasing (should trigger penalty)
action_wrong = AgentAction(utterance="The fee is 499 rupees.")
pen_wrong = instr_det.score(
    State(scenario=scenario, history=[{"role": "agent", "text": "..."}], turn_idx=1),
    action_wrong
)
print(f"Wrong phrasing ('499 rupees'): penalty = {pen_wrong}")

# Test correct phrasing
action_correct = AgentAction(utterance="The fee is rupees four nine nine.")
pen_correct = instr_det.score(
    State(scenario=scenario, history=[{"role": "agent", "text": "..."}], turn_idx=1),
    action_correct
)
print(f"Correct phrasing ('rupees four nine nine'): penalty = {pen_correct}")

print()

# Test 4: Check scenario explicit_rules
print("=" * 60)
print("TEST 4: Check explicit_rules in scenario")
print("=" * 60)

# Load an instruction scenario
obs, state = env.reset(scenario_id="instr_kk_01")
print(f"Scenario {state.scenario.scenario_id}:")
print(f"  explicit_rules: {state.scenario.explicit_rules}")