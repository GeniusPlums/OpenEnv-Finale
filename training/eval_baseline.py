import json
import os
import time
from pathlib import Path
from statistics import mean, stdev

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from role_drift_env.models import AgentAction, Observation, State
from role_drift_env.server.environment import RoleDriftEnvironment
from training.rollout import rollout_episode


def make_groq_policy(model: str, system_prompt: str):
    """Create a policy that uses Groq LLM."""
    from training.groq_client import generate as groq_generate

    def policy(obs, state):
        # Build messages for Groq
        messages = [{"role": "system", "content": obs.system_prompt or "You are a helpful voice agent."}]
        
        # Add conversation history
        for turn in state.history:
            role = "user" if turn["role"] == "customer" else "assistant"
            messages.append({"role": role, "content": turn["text"]})
        
        # Add current customer message
        messages.append({"role": "user", "content": obs.customer_message})
        
        # Generate response from Groq
        text = groq_generate(messages, model=model, max_tokens=60)
        
        # Detect end_call
        end_call = any(kw in text.lower() for kw in ["goodbye", "bye", "end call", "hang up", "see you"])
        
        return AgentAction(utterance=text, end_call=end_call)
    
    return policy


def _make_local_model_policy(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = (
            "{% for message in messages %}{% if message['role'] == 'system' %}"
            "{% set system_message = message['content'] %}{% endif %}{% endfor %}"
            "{% if system_message is defined %}{{ system_message }}{% endif %}"
            "{% for message in messages %}{% if message['role'] != 'system' %}"
            "{% if message['role'] == 'user' %}{{ '\\nUser: ' + message['content'] }}"
            "{% elif message['role'] == 'assistant' %}{{ '\\nAssistant: ' + message['content'] }}"
            "{% endif %}{% endif %}{% endfor %}\\nAssistant:"
        )

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    def policy(obs: Observation, state: State) -> AgentAction:
        messages = [{"role": "system", "content": obs.system_prompt or "You are a helpful voice agent."}]
        for turn in state.history:
            role = "user" if turn["role"] == "customer" else "assistant"
            messages.append({"role": role, "content": turn["text"]})
        messages.append({"role": "user", "content": obs.customer_message})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        max_ctx = getattr(tokenizer, "model_max_length", 2048)
        if max_ctx is None or max_ctx <= 0:
            max_ctx = 2048
        max_prompt_len = max(128, max_ctx - 60 - 1)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        end_call = any(kw in text.lower() for kw in ["goodbye", "bye", "end call", "hang up", "see you"])
        return AgentAction(utterance=text, end_call=end_call)

    return policy


def evaluate_baseline(
    backend: str = None,
    policy_type: str = None,
    model: str = None,
    model_path: str = None,
    scenario_file: str = "data/scenarios/eval.jsonl",
    num_seeds: int = 3,
    max_episodes: int = None,
    output_path: str = None,
):
    """Evaluate a baseline policy against eval scenarios.
    
    Args:
        backend: "heuristic" or "groq"
        policy_type: For heuristic backend: "frontier", "deployable", "sft"
        model: For groq backend: model name like "llama-3.1-8b-instant"
    """
    if output_path is None:
        if model_path:
            output_path = "data/eval_results/model_eval.json"
        elif backend == "groq":
            output_path = f"data/eval_results/groq_{model.replace('-', '_')}.json"
        else:
            output_path = f"eval_{policy_type}.jsonl"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    env = RoleDriftEnvironment()

    # Heuristic policies
    def frontier_like_policy(obs, state):
        if state.turn_idx >= 5:
            return AgentAction(utterance="Thank you for your time. Goodbye!", end_call=True)
        if "workshop" in state.scenario.task_description.lower():
            return AgentAction(utterance="Our workshop covers technical and fundamental analysis.", end_call=False)
        return AgentAction(utterance="I can help you with that. What do you need?", end_call=False)

    def deployable_like_policy(obs, state):
        if state.turn_idx >= 8:
            return AgentAction(utterance="You are welcome. Have a nice day!", end_call=False)
        if "workshop" in state.scenario.task_description.lower():
            return AgentAction(utterance="The fee is 499 rupees. 1. Technical analysis 2. Fundamental analysis", end_call=False)
        return AgentAction(utterance="Let me help you with your application.", end_call=False)

    def sft_policy(obs, state):
        if state.customer_farewell_turn is not None and state.disengagement_counter >= 2:
            return AgentAction(utterance="Goodbye, have a great day!", end_call=True)
        if "workshop" in state.scenario.task_description.lower():
            return AgentAction(utterance="Our workshop covers technical and fundamental analysis.", end_call=False)
        return AgentAction(utterance="I can help you resume your application. Where did you leave off?", end_call=False)

    heuristic_policies = {
        "frontier": frontier_like_policy,
        "deployable": deployable_like_policy,
        "sft": sft_policy,
    }

    if model_path:
        policy = _make_local_model_policy(model_path)
        backend_label = f"model:{model_path}"
    elif backend == "heuristic":
        if policy_type not in heuristic_policies:
            raise ValueError(f"Unknown policy_type: {policy_type}")
        policy = heuristic_policies[policy_type]
        backend_label = policy_type
    elif backend == "groq":
        if not model:
            raise ValueError("--model required for groq backend")
        # Get system prompt from first scenario
        with open(scenario_file, "r") as f:
            first_scenario = json.loads(f.readline())
        env_temp = RoleDriftEnvironment()
        obs_temp, _ = env_temp.reset(scenario_id=first_scenario["scenario_id"])
        policy = make_groq_policy(model, obs_temp.system_prompt)
        backend_label = f"groq_{model}"
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Load scenario IDs
    scenario_ids = []
    with open(scenario_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            scenario_ids.append(obj["scenario_id"])

    results = []
    episode_idx = 0
    for sid in scenario_ids:
        for seed in range(num_seeds):
            if max_episodes is not None and episode_idx >= max_episodes:
                break
            traj, ret = rollout_episode(
                policy=policy,
                scenario_id=sid,
                env=env,
                rollout_idx=seed,
            )
            
            # Collect per-turn rewards
            turn_rewards = []
            for obs, action, reward in traj:
                turn_rewards.append(reward.components)
            
            # Compute per-detector totals
            detector_totals = {}
            for tr in turn_rewards:
                for k, v in tr.items():
                    detector_totals[k] = detector_totals.get(k, 0) + v
            
            results.append({
                "scenario_id": sid,
                "seed": seed,
                "episode_return": ret,
                "turns": len(traj),
                "per_detector": detector_totals,
            })
            print(f"[{backend_label}] {sid} seed={seed} return={ret:.3f} turns={len(traj)}")
            episode_idx += 1
            
            # Rate limit insurance
            if backend == "groq":
                time.sleep(2)
        if max_episodes is not None and episode_idx >= max_episodes:
            break
    if not results:
        raise RuntimeError("No eval episodes were run.")

    returns = [r["episode_return"] for r in results]
    detector_sums = {}
    for row in results:
        for key, value in row["per_detector"].items():
            detector_sums[key] = detector_sums.get(key, 0.0) + value
    per_detector_mean = {k: v / len(results) for k, v in detector_sums.items()}
    summary = {
        "backend": backend_label,
        "scenario_file": scenario_file,
        "num_seeds": num_seeds,
        "num_episodes": len(results),
        "mean_return": mean(returns),
        "std_return": stdev(returns) if len(returns) > 1 else 0.0,
        "per_detector_mean": per_detector_mean,
    }
    payload = {"summary": summary, "episodes": results}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[{backend_label}] Mean return: {summary['mean_return']:.3f} (std={summary['std_return']:.3f})")
    print(f"Results saved to: {output_path}")
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["heuristic", "groq"], help="Optional when --model-path is provided")
    parser.add_argument("--policy", choices=["frontier", "deployable", "sft"], help="For heuristic backend")
    parser.add_argument("--model", help="For groq backend (e.g., llama-3.1-8b-instant, llama-3.3-70b-versatile)")
    parser.add_argument("--model-path", help="Local or HF model path for direct model evaluation")
    parser.add_argument("--scenario-file", default="data/scenarios/eval.jsonl")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--max-episodes", type=int, default=None, help="Cap total episodes across all scenarios and seeds")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    if not args.model_path and not args.backend:
        parser.error("Either --model-path or --backend is required")

    os.environ.setdefault("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))

    evaluate_baseline(
        backend=args.backend,
        policy_type=args.policy,
        model=args.model,
        model_path=args.model_path,
        scenario_file=args.scenario_file,
        num_seeds=args.num_seeds,
        max_episodes=args.max_episodes,
        output_path=args.output,
    )