#!/usr/bin/env python3
"""
Rollout-only inference for diagnostic analysis.
Run from training/ directory: python rollout_diagnostic.py
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, "/app")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from role_drift_env.server.environment import RoleDriftEnvironment
from role_drift_env.server.customer_sim import CustomerSimulator
from role_drift_env.models import AgentAction


def load_scenarios(scenarios_file: str = "data/scenarios/train.jsonl") -> List[Dict]:
    scenarios = []
    with open(scenarios_file, "r", encoding="utf-8") as f:
        for line in f:
            scenarios.append(json.loads(line))
    return scenarios


def setup_model(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=dtype).to(device)
    model.eval()
    return model, tokenizer, device


def generate_action(model, tokenizer, device, obs, state, max_new_tokens=60):
    system = obs.system_prompt or "You are a helpful voice agent."
    messages = [{"role": "system", "content": system}]
    for turn in state.history:
        role = "user" if turn["role"] == "customer" else "assistant"
        messages.append({"role": role, "content": turn["text"]})
    messages.append({"role": "user", "content": obs.customer_message})
    
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"
    
    max_prompt_len = getattr(tokenizer, "model_max_length", 1024)
    if max_prompt_len < 2048:
        max_prompt_len = 2048
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    end_call = any(kw in text.lower() for kw in ["goodbye", "bye", "end call", "hang up", "see you"])
    return AgentAction(utterance=text, end_call=end_call)


def run_rollout(model, tokenizer, device, scenario_id, env, rollout_idx=0):
    obs, state = env.reset(scenario_id=scenario_id, rollout_idx=rollout_idx)
    sim = CustomerSimulator.from_scenario(state.scenario)
    
    turns = []
    done = False
    
    while not done and state.turn_idx < state.scenario.max_turns:
        action = generate_action(model, tokenizer, device, obs, state)
        
        term_raw = env.reward_composer.term_det.score(state, action)
        goal_raw = env.reward_composer.goal_det.score(state, action)
        instr_raw = env.reward_composer.instr_det.score(state, action)
        lang_raw = env.reward_composer.lang_det.score(state, action)
        
        obs_next, reward, done, info = env.step(state, action, sim)
        
        turn_data = {
            "turn_idx": state.turn_idx,
            "customer_message": obs.customer_message,
            "agent_utterance": action.utterance,
            "detector_outputs_raw": {
                "termination": term_raw,
                "goal": goal_raw,
                "instruction": instr_raw,
                "language": lang_raw,
            },
            "reward_components": reward.components,
            "reward_total": reward.total,
            "state_after": {
                "customer_farewell_turn": state.customer_farewell_turn,
                "disengagement_counter": state.disengagement_counter,
                "turn_idx": state.turn_idx,
            }
        }
        turns.append(turn_data)
        
        obs = obs_next
        if state.turn_idx >= state.scenario.max_turns:
            break
    
    terminal = env.check_terminal_success(state)
    
    return {
        "scenario_id": scenario_id,
        "rollout_idx": rollout_idx,
        "task_description": state.scenario.task_description,
        "persona_id": state.scenario.persona_id,
        "drift_types": state.scenario.drift_types,
        "turns": turns,
        "terminal_success": terminal,
        "episode_return": sum(t["reward_total"] for t in turns) + terminal,
    }


def main():
    os.chdir("/app")
    
    all_scenarios = load_scenarios()
    scenario_ids = [s["scenario_id"] for s in all_scenarios[:20]]
    
    print(f"Running rollouts for {len(scenario_ids)} scenarios")
    
    model, tokenizer, device = setup_model()
    env = RoleDriftEnvironment()
    
    results = []
    for i, sid in enumerate(scenario_ids):
        print(f"Processing {i+1}/20: {sid}")
        try:
            result = run_rollout(model, tokenizer, device, sid, env, rollout_idx=i)
            results.append(result)
        except Exception as e:
            print(f"  ERROR on {sid}: {e}")
            import traceback
            traceback.print_exc()
    
    output_path = Path("/app/data/diagnostics/rollout_detector_outputs.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"\nSaved {len(results)} episodes to {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()