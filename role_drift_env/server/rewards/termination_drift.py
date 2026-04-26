import re

from role_drift_env.models import State, AgentAction

# Closing / hang-up cues (agent tries to end before customer has disengaged)
_EXIT_VERBAL = re.compile(
    r"\b(bye|goodbye|hang\s*up|end\s+(the\s+)?call|see\s+you|talk\s+soon|"
    r"gotta\s+go|signing\s+off|have\s+a\s+good\s+day)\b",
    re.IGNORECASE,
)


class TerminationDriftDetector:
    def score(self, state: State, action: AgentAction) -> float:
        """Return penalty magnitude in [0, 1]. 0 means no penalty."""
        parts: list[float] = []

        # --- A) Thank-you loop: agent keeps talking after clear customer farewell ---
        if state.customer_farewell_turn is not None and state.disengagement_counter >= 2:
            turns_since = state.turn_idx - state.customer_farewell_turn
            parts.append(min(0.2 * turns_since, 1.0))

        # --- B) Premature exit: agent ends call before any tracked customer farewell ---
        if state.customer_farewell_turn is None:
            if action.end_call:
                parts.append(0.72)
            else:
                text = (action.utterance or "").strip()
                # Short turns only: avoid "we can say goodbye at the end" false positives
                if text and len(text.split()) <= 18 and _EXIT_VERBAL.search(text):
                    parts.append(0.42)

        # --- C) Incomplete close: scale penalty by how far from full outcome success (ts in [0, 3]) ---
        if action.end_call and state.scenario.outcome_predicates:
            from role_drift_env.server.rewards.terminal_success import compute_terminal_success

            ts = compute_terminal_success(state)
            if ts < 3.0:
                frac_missing = (3.0 - float(ts)) / 3.0
                # exp>1: steep when nearly empty, gentle when almost done (learning slope not a cliff)
                incomplete_pen = 0.72 * (frac_missing**1.25)
                parts.append(incomplete_pen)

        if not parts:
            return 0.0
        return round(min(1.0, sum(parts)), 4)
