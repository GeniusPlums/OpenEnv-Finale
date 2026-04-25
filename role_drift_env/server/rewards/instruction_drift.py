import re
from pathlib import Path
from typing import Dict, List, Any
import json

from role_drift_env.models import State, AgentAction


def _load_rules_for_prompt(prompt_id: str) -> List[Dict]:
    rules_dir = Path("data/prompts/rules")
    rules_file = rules_dir / f"{prompt_id}.json"
    if rules_file.exists():
        with open(rules_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _check_rule(rule: Dict, state: State, action: AgentAction, text: str) -> float:
    rule_type = rule.get("type")
    params = rule.get("params", {})
    penalty = params.get("penalty_on_violation", 0.4)
    text_lower = text.lower()

    if rule_type == "required_words":
        trigger = params.get("trigger_words", [])
        required = params.get("required_words", [])
        if trigger and any(t.lower() in text_lower for t in trigger):
            if required and not any(r.lower() in text_lower for r in required):
                return penalty
        return 0.0

    elif rule_type == "forbidden_words":
        forbidden = params.get("forbidden_words", [])
        for word in forbidden:
            if word.lower() in text_lower:
                return penalty
        return 0.0

    elif rule_type == "max_token_count":
        max_tok = params.get("max_tokens", 100)
        if len(text.split()) > max_tok:
            return penalty
        return 0.0

    elif rule_type == "forbidden_formats":
        if re.search(r"^\s*\d+\.\s", text, re.MULTILINE):
            return penalty
        return 0.0

    return 0.0


class InstructionDriftDetector:
    def __init__(self):
        self.rules_cache: Dict[str, List[Dict]] = {}

    def _get_rules(self, prompt_id: str) -> List[Dict]:
        if prompt_id not in self.rules_cache:
            self.rules_cache[prompt_id] = _load_rules_for_prompt(prompt_id)
        return self.rules_cache[prompt_id]

    def score(self, state: State, action: AgentAction) -> float:
        text = action.utterance
        prompt_id = state.scenario.prompt_id

        penalties = []
        for rule in self._get_rules(prompt_id):
            p = _check_rule(rule, state, action, text)
            if p > 0:
                penalties.append(p)

        for rule in state.scenario.explicit_rules:
            p = _check_rule(rule, state, action, text)
            if p > 0:
                penalties.append(p)

        if penalties:
            return round(min(sum(penalties), 1.0), 4)
        return 0.0