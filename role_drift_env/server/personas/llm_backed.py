import json
from pathlib import Path
from typing import Optional
from role_drift_env.models import State
from .base import Persona

# Try to import vLLM or transformers for local LLM generation
# For now, keep it behind a lazy import so the env can load without a GPU


class LLMPersona(Persona):
    """LLM-backed persona using a local model or API."""

    def __init__(
        self,
        persona_id: str,
        system_prompt: str,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 120,
        drift_targets: list[str] = None,
    ):
        self.persona_id = persona_id
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.drift_targets = drift_targets or []
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            # Lazy import so we don't crash on import if vllm isn't installed
            try:
                from vllm import LLM, SamplingParams
                self._llm = LLM(model=self.model, trust_remote_code=True)
                self._sampling_params = SamplingParams(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception:
                # Fallback: if vLLM isn't available, we'll use a simple template fallback
                self._llm = "fallback"
        return self._llm

    def _build_messages(self, state: State) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        for turn in state.history:
            role = "user" if turn["role"] == "agent" else "assistant"
            messages.append({"role": role, "content": turn["text"]})
        return messages

    def next_utterance(self, state: State, rng_seed: int) -> str:
        llm = self._get_llm()
        if llm == "fallback":
            # Fallback when no LLM is available: scripted-like behavior
            return "Thanks, I think I have what I need. Goodbye."
        from vllm import SamplingParams
        sp = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=rng_seed,
        )
        messages = self._build_messages(state)
        outputs = llm.chat(messages, sampling_params=sp)
        return outputs[0].outputs[0].text.strip()

    def is_farewell(self, utterance: str) -> bool:
        # Delegate to same utility as scripted
        from .scripted import _contains_farewell
        return _contains_farewell(utterance)


def load_llm_persona(persona_id: str, json_path: Optional[Path] = None) -> LLMPersona:
    """Load an LLM persona definition from JSON."""
    if json_path is None:
        json_path = Path("data/personas/adversarial_customers.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    defs = {p["persona_id"]: p for p in data.get("personas", [])}
    if persona_id not in defs:
        raise ValueError(f"LLM persona {persona_id} not found in {json_path}")
    p = defs[persona_id]
    return LLMPersona(
        persona_id=persona_id,
        system_prompt=p["system_prompt"],
        model=p.get("model", "Qwen/Qwen2.5-7B-Instruct"),
        temperature=p.get("temperature", 0.7),
        max_tokens=p.get("max_tokens", 120),
        drift_targets=p.get("drift_targets", []),
    )
