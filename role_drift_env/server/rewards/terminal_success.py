import re
from role_drift_env.models import State, OutcomePredicate


def compute_terminal_success(state: State) -> float:
    """Compute terminal success bonus based on outcome predicates."""
    if not state.scenario.outcome_predicates:
        # Fallback: agent ended call within 10 turns without drifting
        if state.terminated and state.turn_idx <= 10:
            return 3.0
        return 0.0

    hits = []
    full_agent_text = " ".join([t["text"] for t in state.history if t["role"] == "agent"])
    for pred in state.scenario.outcome_predicates:
        hit = _check_predicate(pred, full_agent_text, state)
        hits.append(1.0 if hit else 0.0)
    mean_hit = sum(hits) / len(hits) if hits else 0.0
    return round(mean_hit * 3.0, 4)


def _check_predicate(pred: OutcomePredicate, full_agent_text: str, state: State) -> bool:
    text = full_agent_text.lower()
    patterns = [p.lower() for p in pred.patterns]
    if pred.type == "regex":
        return any(re.search(p, full_agent_text, re.IGNORECASE) for p in pred.patterns)
    elif pred.type == "any_phrase_match":
        return any(p in text for p in patterns)
    elif pred.type == "all_phrase_match":
        return all(p in text for p in patterns)
    elif pred.type == "custom":
        # Custom checkers would be implemented in a separate module
        # For now, fallback to any_phrase_match
        return any(p in text for p in patterns)
    return False
