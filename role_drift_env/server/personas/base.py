from abc import ABC, abstractmethod
from role_drift_env.models import State


class Persona(ABC):
    persona_id: str
    drift_targets: list[str]

    @abstractmethod
    def next_utterance(self, state: State, rng_seed: int) -> str:
        """Return the next customer utterance given the conversation state."""
        ...

    @abstractmethod
    def is_farewell(self, utterance: str) -> bool:
        """Return True if the utterance signals a customer farewell/disengagement."""
        ...
