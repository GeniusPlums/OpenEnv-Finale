import json
import os
import time
from pathlib import Path
from role_drift_env.models import AgentAction
from role_drift_env.server.environment import RoleDriftEnvironment
from role_drift_env.server.customer_sim import CustomerSimulator
from training.rollout import rollout_episode
from training.groq_client import generate as groq_generate


def make_groq_policy(model: str, system_prompt: str):
    """Create a policy that uses Groq LLM."""
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


def evaluate_baseline(
    backend: str,
    policy_type: str = None,
    model: str = None,
    scenario_file: str = "data/scenarios/eval.jsonl",
    num_seeds: int = 3,
    output_path: str = None,
):
    """Evaluate a baseline policy against eval scenarios.
    
    Args:
        backend: "heuristic" or "groq"
        policy_type: For heuristic backend: "frontier", "deployable", "sft"
        model: For groq backend: model name like "llama-3.1-8b-instant"
    """
    if output_path is None:
        if backend == "groq":
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

    if backend == "heuristic":
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
    for sid in scenario_ids:
        for seed in range(num_seeds):
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
            
            # Rate limit insurance
            if backend == "groq":
                time.sleep(2)

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    returns = [r["episode_return"] for r in results]
    print(f"[{backend_label}] Mean return: {sum(returns)/len(returns):.3f}")
    print(f"Results saved to: {output_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["heuristic", "groq"], required=True)
    parser.add_argument("--policy", choices=["frontier", "deployable", "sft"], help="For heuristic backend")
    parser.add_argument("--model", help="For groq backend (e.g., llama-3.1-8b-instant, llama-3.3-70b-versatile)")
    parser.add_argument("--scenario-file", default="data/scenarios/eval.jsonl")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()
    
    os.environ.setdefault("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    
    evaluate_baseline(
        backend=args.backend,
        policy_type=args.policy,
        model=args.model,
        scenario_file=args.scenario_file,
        num_seeds=args.num_seeds,
        output_path=args.output,
    )