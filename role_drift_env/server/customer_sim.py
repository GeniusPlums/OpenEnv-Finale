from role_drift_env.models import State
from .personas import Persona, ScriptedPersona, LLMPersona, get_scripted_persona, load_llm_persona


class CustomerSimulator:
    """Thin wrapper that dispatches to a persona and returns the next customer utterance."""

    def __init__(self, persona: Persona):
        self.persona = persona

    def next_turn(self, state: State, rng_seed: int) -> str:
        return self.persona.next_utterance(state, rng_seed)

    def is_farewell(self, utterance: str) -> bool:
        return self.persona.is_farewell(utterance)

    @staticmethod
    def from_scenario(scenario) -> "CustomerSimulator":
        persona_id = scenario.persona_id
        # Try scripted first, then LLM-backed
        try:
            persona = get_scripted_persona(persona_id)
        except ValueError:
            persona = load_llm_persona(persona_id)
        return CustomerSimulator(persona)
