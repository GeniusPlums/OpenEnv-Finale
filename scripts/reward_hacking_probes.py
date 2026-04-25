"""
Reward Hacking Probes - Validate reward design isn't easily gamed.
Run three trivial policies + prepare for trained checkpoint.
"""
import sys
sys.path.insert(0, "C:/Users/anish/OpenEnv Finale")

import json
import numpy as np
from role_drift_env.server.environment import RoleDriftEnvironment
from role_drift_env.server.rewards import RewardComposer
from role_drift_env.models import AgentAction, State


def run_probe(env, scenario_id, agent_mode, num_seeds=3):
    """Run a probe agent on scenarios, return mean reward."""
    results = []
    for seed in range(num_seeds):
        try:
            obs, state = env.reset(scenario_id, seed)
        except:
            continue
        
        if agent_mode == "silent":
            action = AgentAction(utterance="")
        elif agent_mode == "minimal":
            action = AgentAction(utterance="Okay.")
        elif agent_mode == "verbose":
            action = AgentAction(utterance="Thank you for sharing that information. I understand your interest and will consider the appropriate steps to move forward effectively.")
        else:
            continue  # trained mode needs checkpoint
        
        composer = RewardComposer()
        reward = composer.score(state, action)
        results.append(reward.total)
    
    return np.mean(results) if results else 0.0


def main():
    print("Running reward hacking probes...")
    env = RoleDriftEnvironment()
    
    # Load eval scenarios
    with open("data/scenarios/eval.jsonl", "r") as f:
        scenarios = [json.loads(line)["scenario_id"] for line in f][:10]
    
    modes = ["silent", "minimal", "verbose"]
    results = {}
    
    for mode in modes:
        print(f"Running {mode}...")
        total = 0
        count = 0
        for sid in scenarios:
            r = run_probe(env, sid, mode, num_seeds=2)
            if r != 0:
                total += r
                count += 1
        results[mode] = round(total / max(count, 1), 3)
        print(f"  {mode}: {results[mode]}")
    
    # Save results
    output = {
        "probe_results": results,
        "trained": None,  # placeholder for when compute arrives
    }
    
    with open("data/eval_results/reward_hacking_probe.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults: {results}")
    print("Saved to data/eval_results/reward_hacking_probe.json")
    
    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    labels = ['Silent', 'Minimal', 'Verbose', 'Trained\n(pending)']
    values = [results.get('silent', 0), results.get('minimal', 0), results.get('verbose', 0), 0]
    colors = ['#ff6b6b', '#ffa94d', '#69db7c', '#868e96']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel('Mean Reward')
    ax.set_title('Reward Hacking Probes\n(Trivial policies vs trained)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/reward_hacking_probe.png', dpi=150)
    plt.close()
    print("Plot saved to plots/reward_hacking_probe.png")


if __name__ == "__main__":
    main()