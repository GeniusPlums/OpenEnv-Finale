from dataclasses import dataclass, field
from typing import Literal, Optional

DriftType = Literal["termination", "goal", "instruction", "language", "cooperative"]


@dataclass
class AgentAction:
    utterance: str
    end_call: bool = False       # agent's explicit end-of-call signal


@dataclass
class Observation:
    customer_message: str
    turn_idx: int
    scenario_id: str
    system_prompt: str           # included every step for the trainer's convenience
    done: bool = False


@dataclass
class TurnReward:
    total: float                 # clipped sum used for training
    components: dict[str, float] # per-detector, for logging


@dataclass
class OutcomePredicate:
    predicate_id: str
    type: Literal["regex", "any_phrase_match", "all_phrase_match", "custom"]
    patterns: list[str]          # always a list; a single regex is a one-element list


@dataclass
class Scenario:
    scenario_id: str
    prompt_id: str               # references data/prompts/*
    task_description: str        # short one-liner for goal-drift scoring
    allowed_language: str        # "en", "hi-en", etc.
    persona_id: str              # references data/personas/*
    drift_types: list[DriftType] # scenarios can target multiple
    explicit_rules: list[dict]   # for instruction drift; see Sec. 8.3
    opening_message: str
    outcome_predicates: list[OutcomePredicate]
    max_turns: int = 30
    seed: int = 0                # base seed; rollout_idx is xor'd in


@dataclass
class State:
    scenario: Scenario
    history: list[dict] = field(default_factory=list)  # [{"role": "agent"|"customer", "text": "..."}]
    turn_idx: int = 0
    agent_farewell_turn: Optional[int] = None
    customer_farewell_turn: Optional[int] = None
    terminated: bool = False
    disengagement_counter: int = 0
    last_customer_token_count: int = 0
    outcome_hits: dict = field(default_factory=dict)
