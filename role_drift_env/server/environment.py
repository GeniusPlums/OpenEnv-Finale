import json
import re
from pathlib import Path
from role_drift_env.models import AgentAction, Observation, State, Scenario, TurnReward
from role_drift_env.server.customer_sim import CustomerSimulator
from role_drift_env.server.rewards import RewardComposer
from typing import Tuple, Dict, Any


class RoleDriftEnvironment:
    """OpenEnv-compatible environment for role-drift voice-agent conversations."""

    def __init__(self, scenarios_dir: str = "data/scenarios"):
        self.scenarios_dir = Path(scenarios_dir)
        self.sessions: Dict[str, State] = {}
        self.reward_composer = RewardComposer()

    def _load_scenario(self, scenario_id: str) -> Scenario:
        # Search both train and eval files
        for filename in ["train.jsonl", "eval.jsonl"]:
            path = self.scenarios_dir / filename
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    if obj["scenario_id"] == scenario_id:
                        return self._dict_to_scenario(obj)
        raise ValueError(f"Scenario {scenario_id} not found")

    @staticmethod
    def _dict_to_scenario(obj: dict) -> Scenario:
        from role_drift_env.models import OutcomePredicate
        preds = [OutcomePredicate(**p) for p in obj.get("outcome_predicates", [])]
        return Scenario(
            scenario_id=obj["scenario_id"],
            prompt_id=obj["prompt_id"],
            task_description=obj["task_description"],
            allowed_language=obj["allowed_language"],
            persona_id=obj["persona_id"],
            drift_types=obj["drift_types"],
            explicit_rules=obj.get("explicit_rules", []),
            opening_message=obj["opening_message"],
            outcome_predicates=preds,
            max_turns=obj.get("max_turns", 30),
            seed=obj.get("seed", 0),
        )

    def reset(self, scenario_id: str = None, rollout_idx: int = 0) -> Tuple[Observation, State]:
        if scenario_id is None:
            scenario_id = "termination_001"
        scenario = self._load_scenario(scenario_id)
        state = State(
            scenario=scenario,
            history=[],
            turn_idx=0,
        )
        obs = Observation(
            customer_message=scenario.opening_message,
            turn_idx=0,
            scenario_id=scenario.scenario_id,
            system_prompt=self._load_prompt(scenario.prompt_id),
            done=False,
        )
        return obs, state

    def _load_prompt(self, prompt_id: str) -> str:
        # Map to full production prompts
        prompt_map = {
            "kundan_kishore": "kundan_kishore_full.md",
            "masters_union": "masters_union_full.md",
            "dearconnect": "dearconnect_full.md",
        }
        filename = prompt_map.get(prompt_id, f"{prompt_id}.md")
        path = Path("data/prompts") / filename
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def step(self, state: State, action: AgentAction, sim: CustomerSimulator) -> Tuple[Observation, TurnReward, bool, Dict[str, Any]]:
        # Append agent utterance
        state.history.append({"role": "agent", "text": action.utterance})

        # Score turn reward BEFORE customer replies
        reward = self.reward_composer.score(state, action)

        # Check if agent ended call
        if action.end_call:
            state.terminated = True
            done = True
            info = {"reason": "agent_end_call"}
            obs = Observation(
                customer_message="",
                turn_idx=state.turn_idx + 1,
                scenario_id=state.scenario.scenario_id,
                system_prompt=self._load_prompt(state.scenario.prompt_id),
                done=True,
            )
            return obs, reward, done, info

        # Check max turns BEFORE generating customer reply
        if state.turn_idx >= state.scenario.max_turns - 1:
            done = True
            info = {"reason": "max_turns"}
            obs = Observation(
                customer_message="",
                turn_idx=state.turn_idx + 1,
                scenario_id=state.scenario.scenario_id,
                system_prompt=self._load_prompt(state.scenario.prompt_id),
                done=True,
            )
            return obs, reward, done, info

        # Generate customer reply
        seed = state.scenario.seed ^ state.turn_idx ^ (hash(state.scenario.scenario_id) % 10000)
        customer_msg = sim.next_turn(state, seed)
        state.history.append({"role": "customer", "text": customer_msg})

        # Update farewell/disengagement tracking
        token_count = len(customer_msg.split())
        if sim.is_farewell(customer_msg):
            if state.customer_farewell_turn is None and token_count < 20:
                state.customer_farewell_turn = state.turn_idx
            # Disengagement counter: increments if customer shortens AND farewells
            if state.last_customer_token_count > 0 and token_count <= state.last_customer_token_count:
                state.disengagement_counter += 1
            else:
                state.disengagement_counter = 1 if state.customer_farewell_turn is not None else 0
        else:
            # Reset if not farewell
            state.disengagement_counter = 0

        state.last_customer_token_count = token_count
        state.turn_idx += 1

        done = False
        info = {}

        obs = Observation(
            customer_message=customer_msg,
            turn_idx=state.turn_idx,
            scenario_id=state.scenario.scenario_id,
            system_prompt=self._load_prompt(state.scenario.prompt_id),
            done=done,
        )

        return obs, reward, done, info

    def check_terminal_success(self, state: State) -> float:
        from role_drift_env.server.rewards.terminal_success import compute_terminal_success
        return compute_terminal_success(state)
