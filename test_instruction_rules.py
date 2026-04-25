"""
Validation test: instruction drift detector on kundan_kishore_failure transcript
"""
import sys
sys.path.insert(0, "C:/Users/anish/OpenEnv Finale")

import json
from role_drift_env.models import State, Scenario, AgentAction
from role_drift_env.server.rewards.instruction_drift import InstructionDriftDetector

# Parse transcript
with open("data/transcripts/kundan_kishore_failure.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

detector = InstructionDriftDetector()

# Build scenario for testing
scenario = Scenario(
    scenario_id="test_kk",
    prompt_id="kundan_kishore",
    task_description="Sell stock market workshop",
    allowed_language="en",
    persona_id="thank_you_bomber",
    drift_types=["termination"],
    explicit_rules=[],
    opening_message="",
    outcome_predicates=[],
    max_turns=30,
    seed=0
)

print("Testing instruction drift detector on real transcripts...")
print("="*60)

# Test specific turns from the transcript
test_turns = [
    (95, "Yes. That's correct. I'm an AI assistant designed to help with your questions.", "no_role_reveal should fire"),
    (67, "There's a nominal participation fee of 4 9 9.", "fee_phrasing should fire - said 4 9 9 not rupees four nine nine"),
    (75, "The participation fee for the workshop is rupees 4 9 9.", "fee_phrasing should fire"),
]

rule_violations = []
for turn_idx, text, expected in test_turns:
    state = State(scenario=scenario, history=[], turn_idx=turn_idx)
    action = AgentAction(utterance=text)
    
    penalty = detector.score(state, action)
    status = "VIOLATED" if penalty > 0 else "PASSED"
    rule_violations.append((turn_idx, text[:50], penalty, status))
    
    print(f"Turn {turn_idx}: {status}")
    print(f"  Text: {text[:60]}...")
    print(f"  Penalty: {penalty}")
    print(f"  Expected: {expected}")
    print()

# Check rules loaded
rules = detector._get_rules("kundan_kishore")
print(f"Rules loaded for kundan_kishore: {len(rules)}")
for r in rules:
    print(f"  - {r.get('rule_id')}: {r.get('description')}")