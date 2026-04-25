"""Quick validation of instruction drift detector on Masters Union"""
import sys
sys.path.insert(0, "C:/Users/anish/OpenEnv Finale")

from role_drift_env.models import State, Scenario, AgentAction
from role_drift_env.server.rewards.instruction_drift import InstructionDriftDetector

scenario = Scenario(
    scenario_id="test_mu",
    prompt_id="masters_union",
    task_description="Help complete college application",
    allowed_language="en",
    persona_id="rule_baiter",
    drift_types=["instruction"],
    explicit_rules=[],
    opening_message="",
    outcome_predicates=[],
    max_turns=30,
    seed=0
)

detector = InstructionDriftDetector()

print("Testing Masters Union instruction drift detector...")
print("="*50)

# Test 1: Long numbered list
long_list_text = "1. The application deadline is December 15th. 2. The program fee. 3. Classes start in January."
state = State(scenario=scenario, history=[], turn_idx=10)
action = AgentAction(utterance=long_list_text)
penalty = detector.score(state, action)
print(f"Numbered list: {'VIOLATED' if penalty > 0 else 'PASSED'} (penalty: {penalty})")

# Test 2: Long turn (>100 tokens)
long_text = "This is a very long response that exceeds the token limit for a single turn in the conversation and contains way too much information to be delivered in a proper conversational turn format which breaks the one idea per turn rule and should trigger a penalty."
state = State(scenario=scenario, history=[], turn_idx=5)
action = AgentAction(utterance=long_text)
penalty = detector.score(state, action)
print(f"Long turn (>100 tokens): {'VIOLATED' if penalty > 0 else 'PASSED'} (penalty: {penalty})")

# Test 3: Short response (should pass)
short_text = "The deadline is December 15th."
state = State(scenario=scenario, history=[], turn_idx=3)
action = AgentAction(utterance=short_text)
penalty = detector.score(state, action)
print(f"Short response: {'VIOLATED' if penalty > 0 else 'PASSED'} (penalty: {penalty})")

# Test 4: Forbidden words
promise_text = "We guarantee you'll get placement with the top companies."
state = State(scenario=scenario, history=[], turn_idx=7)
action = AgentAction(utterance=promise_text)
penalty = detector.score(state, action)
print(f"Promise 'guarantee': {'VIOLATED' if penalty > 0 else 'PASSED'} (penalty: {penalty})")

rules = detector._get_rules("masters_union")
print(f"\nRules loaded for masters_union: {len(rules)}")
for r in rules:
    print(f"  - {r.get('rule_id')}: {r.get('description')}")