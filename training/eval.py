import json
import torch
from pathlib import Path
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from role_drift_env.models import AgentAction, Observation, State
from role_drift_env.server.environment import RoleDriftEnvironment
from role_drift_env.server.customer_sim import CustomerSimulator
from training.rollout import rollout_episode


class DummyPolicy:
    """A dummy policy that just says hello and ends call after a few turns."""

    def __init__(self):
        self.turn = 0

    def __call__(self, obs: Observation, state: State) -> AgentAction:
        self.turn += 1
        if self.turn >= 3:
            return AgentAction(utterance="Goodbye!", end_call=True)
        return AgentAction(utterance="Hello, how can I help?", end_call=False)


def evaluate_model(
    model_path: str,
    scenario_file: str = "data/scenarios/eval.jsonl",
    num_seeds: int = 3,
    output_path: str = "eval_results.jsonl",
):
    """Evaluate a model on the eval scenario set."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    def policy(obs: Observation, state: State) -> AgentAction:
        messages = [{"role": "system", "content": obs.system_prompt or "You are a helpful agent."}]
        for turn in state.history:
            role = "user" if turn["role"] == "customer" else "assistant"
            messages.append({"role": role, "content": turn["text"]})
        messages.append({"role": "user", "content": obs.customer_message})
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        end_call = any(kw in text.lower() for kw in ["goodbye", "bye", "end call", "hang up"])
        return AgentAction(utterance=text, end_call=end_call)

    scenario_ids = []
    with open(scenario_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            scenario_ids.append(obj["scenario_id"])

    env = RoleDriftEnvironment()
    results = []
    for sid in scenario_ids:
        for seed in range(num_seeds):
            traj, ret = rollout_episode(
                policy=policy,
                scenario_id=sid,
                env=env,
                rollout_idx=seed,
            )
            results.append({
                "scenario_id": sid,
                "seed": seed,
                "episode_return": ret,
                "turns": len(traj),
            })
            print(f"{sid} seed={seed} return={ret:.3f} turns={len(traj)}")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary
    returns = [r["episode_return"] for r in results]
    print(f"Mean return: {sum(returns)/len(returns):.3f}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--scenario-file", default="data/scenarios/eval.jsonl")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--output", default="eval_results.jsonl")
    args = parser.parse_args()
    evaluate_model(args.model_path, args.scenario_file, args.num_seeds, args.output)
