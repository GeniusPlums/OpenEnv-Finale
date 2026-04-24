#!/usr/bin/env python3
"""Test cases for LanguageDriftDetector - with debug output."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from role_drift_env.models import State, AgentAction, Scenario
from role_drift_env.server.rewards.language_drift import LanguageDriftDetector


def make_state_with_history(history_texts: list, allowed_lang: str = "en"):
    scenario = Scenario(
        scenario_id="test_lang",
        prompt_id="test",
        task_description="Test task",
        allowed_language=allowed_lang,
        persona_id="test",
        drift_types=["language"],
        explicit_rules=[],
        opening_message="Hello",
        outcome_predicates=[],
    )
    history = []
    for i, text in enumerate(history_texts):
        role = "customer" if i % 2 == 0 else "agent"
        history.append({"role": role, "text": text})
    return State(scenario=scenario, history=history, turn_idx=len(history_texts))


def test_english_in_english_conversation():
    history = [
        "Hello, I need help with my application.",
        "I'd be happy to help you with that.",
    ]
    state = make_state_with_history(history)
    detector = LanguageDriftDetector()
    
    action = AgentAction(utterance="Of course, I'd be glad to assist you with your application.")
    score = detector.score(state, action)
    
    assert score == 0.0, f"Expected 0.0, got {score}"
    print("[PASS] Test 1: English in English conversation -> 0.0")


def test_spanish_in_english_conversation():
    history = [
        "Hello, I need help with my application.",
        "I'd be happy to help you with that.",
    ]
    state = make_state_with_history(history)
    detector = LanguageDriftDetector()
    
    action = AgentAction(utterance="Claro! Puedo ayudarte con tu solicitud ahora mismo.")
    score = detector.score(state, action)
    
    print(f"  Spanish in English: score = {score}")
    print(f"  Baseline: {detector._get_baseline_language(state)}")
    print(f"  Agent detected: ", end="")
    try:
        from langdetect import detect
        print(detect(action.utterance))
    except Exception as e:
        print(f"Error: {e}")
    
    assert score > 0.5, f"Expected > 0.5, got {score}"
    print(f"[PASS] Test 2: Spanish in English conversation -> {score} (> 0.5)")


def test_hindi_in_english_conversation():
    history = [
        "Hello, I need help with my application.",
        "I'd be happy to help you with that.",
    ]
    state = make_state_with_history(history)
    detector = LanguageDriftDetector()
    
    action = AgentAction(utterance="Ji haan, main aapki madad kar sakta hoon.")
    score = detector.score(state, action)
    
    print(f"  Hindi in English: score = {score}")
    assert score > 0.5, f"Expected > 0.5, got {score}"
    print(f"[PASS] Test 3: Hindi in English conversation -> {score} (> 0.5)")


def test_empty_string():
    state = make_state_with_history(["Hello"])
    detector = LanguageDriftDetector()
    
    action = AgentAction(utterance="")
    score = detector.score(state, action)
    
    assert score == 0.0, f"Expected 0.0, got {score}"
    print("[PASS] Test 4: Empty string -> 0.0")


def test_single_word():
    state = make_state_with_history(["Hello"])
    detector = LanguageDriftDetector()
    
    action = AgentAction(utterance="ok")
    score = detector.score(state, action)
    
    assert score == 0.0, f"Expected 0.0, got {score}"
    print("[PASS] Test 5: Single word 'ok' -> 0.0")


def test_code_switched():
    history = [
        "Hello, I need help with my application.",
        "I'd be happy to help you with that.",
    ]
    state = make_state_with_history(history)
    detector = LanguageDriftDetector()
    
    action = AgentAction(utterance="Yes please, gracias por tu ayuda!")
    score = detector.score(state, action)
    
    print(f"  Code-switched: score = {score} (documenting)")
    print("[PASS] Test 6: Code-switched (documenting)")


def main():
    print("Running LanguageDriftDetector tests...\n")
    
    test_english_in_english_conversation()
    test_spanish_in_english_conversation()
    test_hindi_in_english_conversation()
    test_empty_string()
    test_single_word()
    test_code_switched()
    
    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    main()