from role_drift_env.models import State, AgentAction


class LanguageDriftDetector:
    def __init__(self):
        self._classifier = None
        self._use_fasttext = False

    def _get_classifier(self):
        if self._classifier is None:
            try:
                import fasttext
                self._classifier = fasttext.load_model("lid.176.bin")
                self._use_fasttext = True
            except Exception:
                try:
                    from langdetect import detect
                    self._classifier = detect
                    self._use_fasttext = False
                except Exception:
                    self._classifier = None
        return self._classifier

    def score(self, state: State, action: AgentAction) -> float:
        """Return penalty magnitude in [0,1] if agent switches language unprompted."""
        allowed = state.scenario.allowed_language
        if allowed == "any":
            return 0.0
        text = action.utterance.strip()
        if not text:
            return 0.0

        # Strip loanwords before classification
        loanwords = self._load_loanwords()
        for lw in loanwords:
            text = text.replace(lw, "")
        text = text.strip()
        if not text:
            return 0.0

        clf = self._get_classifier()
        if clf is None:
            return 0.0

        if self._use_fasttext:
            pred = clf.predict(text.replace("\n", " "), k=1)
            label = pred[0][0].replace("__label__", "")
            conf = pred[1][0]
            if conf > 0.85 and label != allowed:
                return round(min(conf, 1.0), 4)
        else:
            try:
                detected = clf(text)
                if detected != allowed:
                    return 0.5
            except Exception:
                return 0.0
        return 0.0

    def _load_loanwords(self):
        import json
        from pathlib import Path
        path = Path("data/personas/loanwords.json")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return ["namaste", "ji", "haan", "nahi", "achha", "theek", "arre", "yar", "boss", "bhai"]
