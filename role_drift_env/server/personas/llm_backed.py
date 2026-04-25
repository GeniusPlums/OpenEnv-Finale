import json
import os
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
        if self._llm is not None:
            return self._llm
        # OpenAI-compatible vLLM server (single GPU: avoid a second in-process 7B load)
        openai_base = (os.environ.get("ROLE_DRIFT_PERSONA_OPENAI_BASE_URL") or "").strip().rstrip("/")
        if openai_base:
            try:
                from openai import OpenAI

                self._llm = "openai"
                self._openai_base = openai_base
                self._openai_client = OpenAI(
                    base_url=openai_base,
                    api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
                )
            except Exception as e:
                print(f"[LLMPersona] OpenAI path failed, falling back to in-process vLLM or fallback: {e}")
                self._openai_client = None
                self._llm = None
            if self._llm == "openai":
                return self._llm
        if self._llm is None:
            # Lazy import so we don't crash on import if vllm isn't installed
            try:
                from vllm import LLM, SamplingParams
                # Keep defaults conservative so persona-vLLM can coexist with policy model.
                gpu_mem_util = float(os.getenv("ROLE_DRIFT_VLLM_GPU_UTIL", "0.35"))
                max_model_len = int(os.getenv("ROLE_DRIFT_VLLM_MAX_MODEL_LEN", "2048"))
                self._llm = LLM(
                    model=self.model,
                    trust_remote_code=True,
                    gpu_memory_utilization=gpu_mem_util,
                    max_model_len=max_model_len,
                )
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
        messages = self._build_messages(state)
        if llm == "openai":
            oai = getattr(self, "_openai_client", None)
            if oai is None:
                return "Thanks, I think I have what I need. Goodbye."
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            try:
                r = oai.chat.completions.create(**kwargs, seed=rng_seed)
            except TypeError:
                r = oai.chat.completions.create(**kwargs)
            return (r.choices[0].message.content or "").strip()
        from vllm import SamplingParams
        sp = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=rng_seed,
        )
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
