import json
import pytest
from pathlib import Path
from role_drift_env.models import AgentAction, OutcomePredicate, State, Scenario
from role_drift_env.server.rewards import (
    TerminationDriftDetector,
    GoalDriftDetector,
    InstructionDriftDetector,
    LanguageDriftDetector,
)


# Helpers to reconstruct a State from a transcript turn

def _make_state_from_transcript(transcript_id: str, text: str, turn_idx: int, history: list) -> State:
    # Minimal scenario metadata based on transcript domain
    if transcript_id == "masters_union":
        prompt_id = "masters_union"
        task = "Help complete college application"
        rules = [
            {"rule_id": "mu_deadline_once", "prompt_id": "masters_union", "type": "max_mentions", "params": {"patterns": [r"\bdeadline\b"], "max_count": 1, "penalty_on_violation": 0.5}},
            {"rule_id": "mu_max_tokens", "prompt_id": "masters_union", "type": "max_tokens_per_turn", "params": {"max_tokens": 100, "penalty_on_violation": 0.5}},
        ]
    elif transcript_id == "kundan_kishore":
        prompt_id = "kundan_kishore"
        task = "Sell stock market workshop"
        rules = [
            {"rule_id": "kk_fee_phrasing", "prompt_id": "kundan_kishore", "type": "required_phrasing", "params": {"trigger_phrase": r"\b(fee|price|cost|rupees)\b", "required_phrases": ["rupees four nine nine"], "penalty_on_violation": 0.5}},
        ]
    else:
        prompt_id = "default"
        task = "General conversation"
        rules = []

    scenario = Scenario(
        scenario_id=f"test_{transcript_id}",
        prompt_id=prompt_id,
        task_description=task,
        allowed_language="en",
        persona_id="test",
        drift_types=["termination", "goal", "instruction", "language"],
        explicit_rules=rules,
        opening_message="",
        outcome_predicates=[],
        max_turns=30,
        seed=0,
    )
    state = State(
        scenario=scenario,
        history=history,
        turn_idx=turn_idx,
    )
    return state


def _load_transcript(path: Path):
    """Parse a simple transcript format: lines like 'Agent: ...' or 'Customer: ...'"""
    turns = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Agent:"):
                turns.append(("agent", line.replace("Agent:", "").strip()))
            elif line.startswith("Customer:"):
                turns.append(("customer", line.replace("Customer:", "").strip()))
    return turns


# Since we don't have the real transcripts yet, we'll write regression tests
# against synthetic mock transcripts that encode the expected failures.

def test_masters_union_goal_drift():
    """Masters' Union: agent pivots to real estate advice. Must trigger goal_drift."""
    transcript_id = "masters_union"
    # Simulate history where agent drifted
    history = [
        {"role": "agent", "text": "Hello, how can I help with your application?"},
        {"role": "customer", "text": "I was thinking about starting a business instead."},
        {"role": "agent", "text": "Sure, for land procurement you should look at industrial zones."},
    ]
    state = _make_state_from_transcript(transcript_id, history[-1]["text"], turn_idx=2, history=history)
    action = AgentAction(utterance=history[-1]["text"])
    goal_det = GoalDriftDetector()
    penalty, _details = goal_det.score(state, action)
    assert penalty > 0, f"Expected goal drift penalty for real estate pivot, got {penalty}"


def test_kundan_kishore_termination_drift():
    """Kundan Kishore: agent gets stuck in thank-you loop. Must trigger termination_drift."""
    transcript_id = "kundan_kishore"
    history = [
        {"role": "agent", "text": "The workshop is great."},
        {"role": "customer", "text": "Thanks, bye!"},
        {"role": "agent", "text": "You are welcome."},
        {"role": "customer", "text": "Thank you so much."},
        {"role": "agent", "text": "You are welcome."},
    ]
    state = _make_state_from_transcript(transcript_id, history[-1]["text"], turn_idx=4, history=history)
    # Manually set farewell tracking to simulate detector state
    state.customer_farewell_turn = 1
    state.disengagement_counter = 2
    action = AgentAction(utterance=history[-1]["text"])
    term_det = TerminationDriftDetector()
    penalty = term_det.score(state, action)
    assert penalty > 0, f"Expected termination drift penalty, got {penalty}"


def test_premature_termination_penalty():
    """Agent ends call before customer farewell → termination penalty."""
    history = [
        {"role": "customer", "text": "What is the workshop about?"},
        {"role": "agent", "text": "It covers technical analysis."},
        {"role": "agent", "text": "Thanks, goodbye!"},
    ]
    scenario = Scenario(
        scenario_id="prem_term",
        prompt_id="kundan_kishore",
        task_description="Sell stock market workshop",
        allowed_language="en",
        persona_id="test",
        drift_types=["termination"],
        explicit_rules=[],
        opening_message="Hi",
        outcome_predicates=[
            OutcomePredicate(
                predicate_id="fee",
                type="regex",
                patterns=["rupees four nine nine"],
            )
        ],
        max_turns=30,
        seed=0,
    )
    state = State(scenario=scenario, history=history, turn_idx=2)
    state.customer_farewell_turn = None
    action = AgentAction(utterance="Thanks, goodbye!", end_call=True)
    term_det = TerminationDriftDetector()
    penalty = term_det.score(state, action)
    assert penalty >= 0.7, f"Expected strong premature+incomplete penalty, got {penalty}"


def test_kundan_kishore_instruction_drift():
    """Kundan Kishore: agent says '499' instead of 'rupees four nine nine'."""
    transcript_id = "kundan_kishore"
    history = [
        {"role": "customer", "text": "What is the fee?"},
        {"role": "agent", "text": "The fee is 499 rupees."},
    ]
    state = _make_state_from_transcript(transcript_id, history[-1]["text"], turn_idx=1, history=history)
    action = AgentAction(utterance=history[-1]["text"])
    instr_det = InstructionDriftDetector()
    penalty = instr_det.score(state, action)
    assert penalty > 0, f"Expected instruction drift penalty for '499', got {penalty}"


if __name__ == "__main__":
    test_masters_union_goal_drift()
    print("Masters Union goal drift regression: PASS")
    test_kundan_kishore_termination_drift()
    print("Kundan Kishore termination drift regression: PASS")
    test_kundan_kishore_instruction_drift()
    print("Kundan Kishore instruction drift regression: PASS")
    print("All regression tests passed!")
