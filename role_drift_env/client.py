import requests
from role_drift_env.models import AgentAction, Observation, TurnReward


class EnvClient:
    """OpenEnv-compatible client for the Role Drift Environment."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session_id: str | None = None

    def reset(self, scenario_id: str = None, **kwargs) -> Observation:
        url = f"{self.base_url}/reset"
        payload = {"scenario_id": scenario_id} if scenario_id else {}
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data["session_id"]
        return Observation(**data["observation"])

    def step(self, action: AgentAction) -> tuple[Observation, TurnReward, bool, dict]:
        if self.session_id is None:
            raise RuntimeError("Call reset() before step()")
        url = f"{self.base_url}/step"
        payload = {
            "session_id": self.session_id,
            "action": {
                "utterance": action.utterance,
                "end_call": action.end_call,
            },
        }
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        obs = Observation(**data["observation"])
        reward = TurnReward(**data["reward"])
        done = data["done"]
        info = data.get("info", {})
        if done:
            self.session_id = None
        return obs, reward, done, info

    def state(self):
        if self.session_id is None:
            raise RuntimeError("No active session")
        url = f"{self.base_url}/state"
        resp = requests.get(url, params={"session_id": self.session_id})
        resp.raise_for_status()
        return resp.json()
