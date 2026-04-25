"""2-group bar: in-domain vs transfer aggregate (baseline vs trained)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    paths = {
        ("In-domain", "Baseline"): Path("data/eval_results/baseline_qwen_1_5b.json"),
        ("In-domain", "Trained"): Path("data/eval_results/trained_qwen_1_5b.json"),
        ("Transfer (DearConnect)", "Baseline"): Path("data/eval_results/baseline_qwen_1_5b_transfer.json"),
        ("Transfer (DearConnect)", "Trained"): Path("data/eval_results/trained_qwen_1_5b_transfer.json"),
    }
    data = {}
    for k, p in paths.items():
        if not p.is_file():
            print(f"Missing: {p}")
            return
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        data[k] = d["summary"].get("mean_return", 0.0)

    categories = ["In-domain", "Transfer (DearConnect)"]
    baseline_vals = [data[(c, "Baseline")] for c in categories]
    trained_vals = [data[(c, "Trained")] for c in categories]

    x = range(len(categories))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar([i - w / 2 for i in x], baseline_vals, w, label="Prompted", color="#888")
    ax.bar([i + w / 2 for i in x], trained_vals, w, label="Trained", color="#2a9d8f")
    ax.set_ylabel("Aggregate mean (episode return)")
    ax.set_title("In-Domain vs Transfer (DearConnect)")
    ax.set_xticks(list(x), categories)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    Path("plots").mkdir(parents=True, exist_ok=True)
    out = Path("plots/in_domain_vs_transfer.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
