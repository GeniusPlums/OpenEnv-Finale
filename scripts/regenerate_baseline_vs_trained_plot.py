"""
Headline bar chart: in-domain baseline vs trained from eval_result JSONs.
Expects `evaluate_baseline` schema: summary.mean_return, summary.per_detector_mean, episodes[*].episode_return
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def bootstrap_mean_ci(
    values: list[float], n_bootstrap: int = 2000, seed: int = 42, ci: float = 0.95
) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    a = np.array(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(a, size=a.size, replace=True).mean()) for _ in range(n_bootstrap)]
    lo, hi = np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    m = float(a.mean())
    return m, m - lo, hi - m


def load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    base_p = Path("data/eval_results/baseline_qwen_1_5b.json")
    train_p = Path("data/eval_results/trained_qwen_1_5b.json")
    if not base_p.is_file() or not train_p.is_file():
        print("Need both:")
        print(f"  {base_p}")
        print(f"  {train_p}")
        return

    base = load(base_p)
    tr = load(train_p)

    det_keys = ["term", "goal", "instr", "lang"]
    labels = ["Termination", "Goal", "Instruction", "Language", "Aggregate"]
    b_vals, t_vals, b_yerr, t_yerr = [], [], [], []

    for k in det_keys:
        b_ep = [e["per_detector"].get(k, 0.0) for e in base.get("episodes", [])]
        t_ep = [e["per_detector"].get(k, 0.0) for e in tr.get("episodes", [])]
        mb, e_lo, e_hi = bootstrap_mean_ci(b_ep)
        mt, e2_lo, e2_hi = bootstrap_mean_ci(t_ep)
        b_vals.append(mb)
        t_vals.append(mt)
        b_yerr.append((e_lo + e_hi) / 2)
        t_yerr.append((e2_lo + e2_hi) / 2)

    r_b = [e["episode_return"] for e in base.get("episodes", [])]
    r_t = [e["episode_return"] for e in tr.get("episodes", [])]
    ma, a_lo, a_hi = bootstrap_mean_ci(r_b)
    mb, b_lo, b_hi = bootstrap_mean_ci(r_t)
    b_vals.append(ma)
    t_vals.append(mb)
    b_yerr.append((a_lo + a_hi) / 2)
    t_yerr.append((b_lo + b_hi) / 2)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x - width / 2,
        b_vals,
        width,
        yerr=b_yerr,
        label="Prompted Qwen 1.5B",
        color="#888",
        capsize=4,
    )
    ax.bar(
        x + width / 2,
        t_vals,
        width,
        yerr=t_yerr,
        label="Trained Qwen 1.5B (GRPO)",
        color="#2a9d8f",
        capsize=4,
    )
    ax.set_ylabel("Per-episode value (higher is better)")
    ax.set_title("Baseline vs Trained — In-Domain (eval set)")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    Path("plots").mkdir(parents=True, exist_ok=True)
    out = Path("plots/baseline_vs_trained.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")
    print(
        f"Aggregate mean: baseline={ma:.3f} trained={mb:.3f} delta={mb - ma:+.3f} (95% CI via bootstrap of episodes)"
    )


if __name__ == "__main__":
    main()
