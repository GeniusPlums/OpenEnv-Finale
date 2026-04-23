from role_drift_env.models import State, AgentAction


class TerminationDriftDetector:
    def score(self, state: State, action: AgentAction) -> float:
        """Return penalty magnitude in [0,1]. 0 means no penalty."""
        if state.customer_farewell_turn is None:
            return 0.0
        if state.disengagement_counter < 2:
            return 0.0
        turns_since = state.turn_idx - state.customer_farewell_turn
        penalty = min(0.2 * turns_since, 1.0)
        return round(penalty, 4)
