import re
from role_drift_env.models import State, AgentAction


class InstructionDriftDetector:
    def score(self, state: State, action: AgentAction) -> float:
        """Return penalty magnitude in [0,1] for instruction drift violations."""
        penalties = []
        text = action.utterance
        for rule in state.scenario.explicit_rules:
            rule_type = rule.get("type")
            params = rule.get("params", {})
            if rule_type == "max_mentions":
                patterns = params.get("patterns", [])
                max_count = params.get("max_count", 1)
                count = 0
                for pat in patterns:
                    count += len(re.findall(pat, text, re.IGNORECASE))
                if count > max_count:
                    penalties.append(params.get("penalty_on_violation", 0.5))
            elif rule_type == "required_phrasing":
                trigger = params.get("trigger_phrase", "")
                if trigger and re.search(trigger, text, re.IGNORECASE):
                    required = params.get("required_phrases", [])
                    if not any(re.search(rp, text, re.IGNORECASE) for rp in required):
                        penalties.append(params.get("penalty_on_violation", 0.5))
            elif rule_type == "forbidden_format":
                pattern = params.get("pattern", "")
                if pattern and re.search(pattern, text, re.MULTILINE):
                    penalties.append(params.get("penalty_on_violation", 0.5))
            elif rule_type == "max_tokens_per_turn":
                max_tok = params.get("max_tokens", 100)
                if len(text.split()) > max_tok:
                    penalties.append(params.get("penalty_on_violation", 0.5))
        if penalties:
            return round(min(sum(penalties), 1.0), 4)
        return 0.0
