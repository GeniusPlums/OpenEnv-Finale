from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uuid
from role_drift_env.models import AgentAction, Observation, State, TurnReward
from role_drift_env.server.environment import RoleDriftEnvironment
from role_drift_env.server.customer_sim import CustomerSimulator

app = FastAPI(title="Role Drift Environment")

# Global env instance and session store
env = RoleDriftEnvironment()
sessions: dict[str, tuple[State, CustomerSimulator]] = {}


class ResetRequest(BaseModel):
    scenario_id: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    action: dict  # AgentAction fields


@app.post("/reset")
def reset(req: ResetRequest):
    scenario_id = req.scenario_id or "termination_001"
    session_id = str(uuid.uuid4())
    obs, state = env.reset(scenario_id=scenario_id)
    sim = CustomerSimulator.from_scenario(state.scenario)
    sessions[session_id] = (state, sim)
    return {"session_id": session_id, "observation": obs}


@app.post("/step")
def step(req: StepRequest):
    session_id = req.session_id
    if session_id not in sessions:
        return {"error": "session not found"}, 404
    state, sim = sessions[session_id]
    action = AgentAction(**req.action)
    obs, reward, done, info = env.step(state, action, sim)
    # Update session state
    sessions[session_id] = (state, sim)
    if done:
        # Terminal success
        terminal = env.check_terminal_success(state)
        info["terminal_success"] = terminal
        # Clean up session
        del sessions[session_id]
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def get_state(session_id: str):
    if session_id not in sessions:
        return {"error": "session not found"}, 404
    state, _ = sessions[session_id]
    return state
