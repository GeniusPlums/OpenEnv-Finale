from .termination_drift import TerminationDriftDetector
from .goal_drift import GoalDriftDetector
from .instruction_drift import InstructionDriftDetector
from .language_drift import LanguageDriftDetector
from .terminal_success import compute_terminal_success
from role_drift_env.models import State, AgentAction, TurnReward


class RewardComposer:
    """Composes per-turn rewards from all drift detectors and task bonus."""

    def __init__(self, weights=None):
        if weights is None:
            weights = {
                "task": 1.0,
                "term": 0.5,
                "goal": 0.5,
                "instr": 0.5,
                "lang": 0.3,
            }
        self.weights = weights
        self.term_det = TerminationDriftDetector()
        self.goal_det = GoalDriftDetector()
        self.instr_det = InstructionDriftDetector()
        self.lang_det = LanguageDriftDetector()

    def score(self, state: State, action: AgentAction) -> TurnReward:
        components = {}

        # Task bonus: small positive for any non-empty, non-drifted turn
        task_bonus = 0.0
        if action.utterance.strip() and len(action.utterance.split()) <= 80:
            # Check if any detector flags this turn
            term_pen = self.term_det.score(state, action)
            goal_pen = self.goal_det.score(state, action)
            instr_pen = self.instr_det.score(state, action)
            lang_pen = self.lang_det.score(state, action)
            if term_pen == 0 and goal_pen == 0 and instr_pen == 0 and lang_pen == 0:
                task_bonus = 0.1
        components["task"] = round(task_bonus, 4)

        # Termination drift
        term_penalty = self.term_det.score(state, action)
        components["term"] = round(-self.weights["term"] * term_penalty, 4)

        # Goal drift
        goal_penalty = self.goal_det.score(state, action)
        components["goal"] = round(-self.weights["goal"] * goal_penalty, 4)

        # Instruction drift
        instr_penalty = self.instr_det.score(state, action)
        components["instr"] = round(-self.weights["instr"] * instr_penalty, 4)

        # Language drift
        lang_penalty = self.lang_det.score(state, action)
        components["lang"] = round(-self.weights["lang"] * lang_penalty, 4)

        total = sum(components.values())
        # Clip total to reasonable range
        total = max(-5.0, min(5.0, total))
        return TurnReward(total=round(total, 4), components=components)
