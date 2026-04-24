from role_drift_env.models import State, AgentAction
from sentence_transformers import SentenceTransformer
import torch


# Threshold set at 0.18 based on histogram valley analysis from diagnostic 
# run on 2026-04-24. See scripts/detector_diagnostic_report.md.
# Off-topic mode: 0-0.15. On-topic mode: 0.70+. Valley min at 0.18.
# Boundary analysis showed:
# - Score 0.15: "Flipping houses provides better returns than trading" (arguably on-topic)
# - Score 0.21: "The price for the course is Rs. 499" (clearly on-topic)
# Using 0.18 as threshold to split these boundary cases.
GOAL_DRIFT_THRESHOLD = 0.18


class GoalDriftDetector:
    _model = None

    def __init__(self):
        if GoalDriftDetector._model is None:
            GoalDriftDetector._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.model = GoalDriftDetector._model

    def score(self, state: State, action: AgentAction) -> float:
        """Return penalty magnitude in [0,1] based on embedding similarity to task description."""
        task_desc = state.scenario.task_description.strip()
        if not task_desc:
            return 0.0
        agent_text = action.utterance.strip()
        if not agent_text:
            return 0.0
        # Embed both
        embeddings = self.model.encode([agent_text, task_desc], convert_to_tensor=True)
        sim = torch.nn.functional.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()
        # Penalty if similarity is below threshold
        if sim < GOAL_DRIFT_THRESHOLD:
            penalty = (GOAL_DRIFT_THRESHOLD - sim) / GOAL_DRIFT_THRESHOLD
            return round(min(penalty, 1.0), 4)
        return 0.0
