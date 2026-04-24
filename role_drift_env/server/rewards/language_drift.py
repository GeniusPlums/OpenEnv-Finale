from role_drift_env.models import State, AgentAction


class LanguageDriftDetector:
    def __init__(self):
        self._classifier = None
        self._baseline_language = None

    def _get_classifier(self):
        if self._classifier is None:
            try:
                from langdetect import detect, detect_langs
                self._classifier = detect
                self._detect_langs = detect_langs
            except Exception:
                self._classifier = None
                self._detect_langs = None
        return self._classifier

    def score(self, state: State, action: AgentAction) -> float:
        """Return penalty magnitude in [0,1] if agent switches language unprompted.
        
        Compares agent's response language to the baseline language established
        by the customer's opening message. This allows for scenarios where the
        customer starts in a non-English language - the agent should match.
        """
        text = action.utterance.strip()
        if not text:
            return 0.0
        
        # Too short to detect reliably
        if len(text.split()) < 3:
            return 0.0
        
        # Strip loanwords before classification (common code-switching)
        loanwords = self._load_loanwords()
        for lw in loanwords:
            text = text.replace(lw, "")
        text = text.strip()
        
        if len(text.split()) < 3:
            return 0.0
        
        clf = self._get_classifier()
        if clf is None:
            return 0.0
        
        try:
            # Get baseline language from customer's first message
            baseline_lang = self._get_baseline_language(state)
            if baseline_lang is None:
                return 0.0
            
            # Detect agent's language
            agent_lang = clf(text)
            
            # Same language as baseline - no penalty
            if agent_lang == baseline_lang:
                return 0.0
            
            # Different language - compute confidence-based penalty
            try:
                lang_probs = self._detect_langs(text)
                if lang_probs:
                    # Use probability of detected language as confidence
                    conf = lang_probs[0].prob
                    return round(min(conf, 1.0), 4)
            except Exception:
                pass
            
            # Fallback: if we know it's different but can't get confidence
            return 0.5
            
        except Exception:
            return 0.0

    def _get_baseline_language(self, state: State) -> str:
        """Detect language from customer's first message in the conversation.
        
        Checks: history (if customer has spoken), then scenario opening message.
        """
        clf = self._get_classifier()
        if clf is None:
            return None
        
        # First, try to get from conversation history
        for turn in state.history:
            if turn.get("role") == "customer":
                text = turn.get("text", "").strip()
                if text and len(text.split()) >= 2:
                    try:
                        return clf(text)
                    except Exception:
                        pass
        
        # Fallback: check scenario opening message
        opening = state.scenario.opening_message
        if opening and len(opening.split()) >= 2:
            try:
                return clf(opening)
            except Exception:
                pass
        
        return None

    def _load_loanwords(self):
        import json
        from pathlib import Path
        path = Path("data/personas/loanwords.json")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return ["namaste", "ji", "haan", "nahi", "achha", "theek", "arre", "yar", "boss", "bhai"]
