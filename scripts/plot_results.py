import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def plot_reward_curve(log_path: str = "checkpoints/grpo/episode_log.jsonl", output_path: str = "plots/reward_curve.png"):
    """Plot reward curve from GRPO episode log."""
    path = Path(log_path)
    if not path.exists():
        print(f"Log file not found: {path}")
        return

    episodes = []
    mean_returns = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            episodes.append(obj["episode"])
            mean_returns.append(obj["mean_return"])

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, mean_returns, label="Mean Return")
    # Rolling mean
    window = 20
    if len(mean_returns) >= window:
        rolling = np.convolve(mean_returns, np.ones(window)/window, mode='valid')
        plt.plot(episodes[window-1:], rolling, label=f"Rolling Mean ({window})", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Mean Episode Return")
    plt.title("GRPO Training Reward Curve")
    plt.legend()
    plt.grid(True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved reward curve to {output_path}")


def plot_eval_comparison(eval_paths: dict, output_path: str = "plots/eval_comparison.png"):
    """Plot eval comparison across models."""
    models = []
    means = []
    for model_name, path in eval_paths.items():
        if not Path(path).exists():
            continue
        returns = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                returns.append(obj["episode_return"])
        models.append(model_name)
        means.append(np.mean(returns) if returns else 0)

    if not models:
        print("No eval data found")
        return

    plt.figure(figsize=(8, 6))
    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
    bars = plt.bar(models, means, color=colors[:len(models)])
    plt.ylabel("Mean Episode Return")
    plt.title("Eval Comparison: Mean Episode Return by Model")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y")
    for bar, val in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{val:.2f}", ha="center", va="bottom")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved eval comparison to {output_path}")


if __name__ == "__main__":
    plot_reward_curve()
    plot_eval_comparison({
        "Frontier (GPT-4o)": "eval_frontier.jsonl",
        "Deployable (Llama 3.1 8B)": "eval_deployable.jsonl",
        "SFT-only (Qwen 1.5B)": "eval_sft.jsonl",
        "GRPO (Qwen 1.5B)": "eval_grpo.jsonl",
    })
