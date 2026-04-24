import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from role_drift_env.models import State, Scenario, AgentAction
from role_drift_env.server.rewards.instruction_drift import InstructionDriftDetector

# Create a proper scenario WITH explicit_rules from train.jsonl
scenario = Scenario(
    scenario_id="instr_kk_01",
    prompt_id="kundan_kishore",
    task_description="Sell stock market workshop",
    allowed_language="en",
    persona_id="rule_baiter",
    drift_types=["instruction"],
    explicit_rules=[
        {
            "rule_id": "kk_fee_phrasing",
            "prompt_id": "kundan_kishore",
            "type": "required_phrasing",
            "params": {
                "trigger_phrase": r"\b(fee|price|cost|rupees)\b",
                "required_phrases": ["rupees four nine nine"],
                "penalty_on_violation": 0.5
            }
        },
        {
            "rule_id": "kk_no_numbered_list",
            "prompt_id": "kundan_kishore",
            "type": "forbidden_format",
            "params": {
                "pattern": r"^\s*\d+\.\s",
                "penalty_on_violation": 0.5
            }
        }
    ],
    opening_message="Tell me everything: fee, schedule, curriculum, faculty, and outcomes.",
    outcome_predicates=[],
    max_turns=30,
    seed=50,
)

instr_det = InstructionDriftDetector()

# Test wrong phrasing (should trigger penalty)
action_wrong = AgentAction(utterance="The fee is 499 rupees.")
state_wrong = State(scenario=scenario, history=[{"role": "agent", "text": "..."}], turn_idx=1)
pen_wrong = instr_det.score(state_wrong, action_wrong)
print(f"Wrong phrasing ('499 rupees'): penalty = {pen_wrong}")

# Test correct phrasing (should NOT trigger)
action_correct = AgentAction(utterance="The fee is rupees four nine nine.")
state_correct = State(scenario=scenario, history=[{"role": "agent", "text": "..."}], turn_idx=1)
pen_correct = instr_det.score(state_correct, action_correct)
print(f"Correct phrasing ('rupees four nine nine'): penalty = {pen_correct}")

# Test numbered list (should trigger forbidden_format)
action_list = AgentAction(utterance="Here are the steps: 1. Sign up 2. Pay 3. Start learning")
state_list = State(scenario=scenario, history=[{"role": "agent", "text": "..."}], turn_idx=1)
pen_list = instr_det.score(state_list, action_list)
print(f"Numbered list (forbidden): penalty = {pen_list}")