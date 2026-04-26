"""
Generate V10 plots: training curves, per-drift learning, in-domain and transfer bar charts.
Reads local data/eval_results and data/training_logs (or env vars).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Bundle B defaults: match hf download paths under repo root
EVAL_DIR = os.environ.get("V10_EVAL_DIR", "data/eval_results")
EPISODE_LOG = os.environ.get("V10_EPISODE_LOG", "data/training_logs/run_v9/episode_log.jsonl")


def _resolve_repo_path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (REPO / p)


DATA_EVAL = _resolve_repo_path(EVAL_DIR)
TRAIN_LOG = _resolve_repo_path(EPISODE_LOG)
SCEN_TRAIN = REPO / "data" / "scenarios" / "train.jsonl"


def _load_scenario_drift() -> dict[str, str]:
    m: dict[str, str] = {}
    if not SCEN_TRAIN.is_file():
        return m
    for line in SCEN_TRAIN.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        tid = o.get("drift_types") or []
        p = next((t for t in tid if t != "cooperative"), None) or (tid[0] if tid else "coop")
        m[o["scenario_id"]] = p
    return m


def plot_reward_curve(out_dir: Path) -> None:
    p = Path(TRAIN_LOG)
    if not p.is_file():
        print(f"skip reward_curve: missing {p}", file=sys.stderr)
        return
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        return
    ep = np.array([r["episode"] for r in rows], dtype=int)
    mean_ret = np.array([r["mean_return"] for r in rows], dtype=float)
    std_ret = np.array([r["std_return"] for r in rows], dtype=float)
    w = 10
    roll = np.convolve(mean_ret, np.ones(w) / w, mode="valid")
    roll_x = np.arange(w - 1, len(mean_ret))
    best_i = int(np.argmax(mean_ret))
    best_v = float(mean_ret[best_i])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(ep, mean_ret, alpha=0.4, label="mean_return (per ep)")
    ax.fill_between(
        ep, mean_ret - std_ret, mean_ret + std_ret, alpha=0.2, color="C0", label="±std"
    )
    if len(roll) > 0:
        ax.plot(roll_x, roll, color="C1", linewidth=2, label=f"rolling {w} mean")
    ax.axvline(16, color="red", alpha=0.4, linestyle="--", label="ep 16–17: KL check")
    ax.axvline(17, color="red", alpha=0.4, linestyle="--")
    ax.annotate(
        f"best ep={best_i}, {best_v:.3f}",
        xy=(best_i, best_v),
        xytext=(best_i, best_v + 0.1),
        arrowprops=dict(arrowstyle="->", color="gray"),
    )
    ax.set_title("GRPO training reward, 100 episodes, Qwen2.5-1.5B")
    ax.set_xlabel("episode")
    ax.set_ylabel("group mean return (within GRPO group)")
    ax.legend(loc="lower right", fontsize=8)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "reward_curve.png", dpi=120)
    plt.close(fig)


def plot_per_drift_curve(out_dir: Path) -> None:
    p = Path(TRAIN_LOG)
    dmap = _load_scenario_drift()
    if not p.is_file() or not dmap:
        print(f"skip per_drift: log={p.is_file()} train_map={bool(dmap)}", file=sys.stderr)
        return
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    key_order = ["termination", "goal", "instruction", "language"]
    for i, dkey in enumerate(key_order):
        ax = axs.flat[i]
        xs = []
        ys = []
        for r in rows:
            sid = r.get("scenario_id")
            dt = dmap.get(sid, "cooperative")
            if dt != dkey:
                continue
            xs.append(r["episode"])
            ys.append(r["mean_return"])
        if xs:
            ax.scatter(xs, ys, s=12, alpha=0.5)
            order = np.argsort(np.array(xs))
            xs_a = np.array(xs)[order]
            ys_a = np.array(ys)[order]
            w = 5
            if len(ys_a) >= w:
                sm = np.convolve(ys_a, np.ones(w) / w, mode="valid")
                ax.plot(
                    xs_a[w - 1 :],
                    sm,
                    color="C1",
                    linewidth=2,
                    label="smoothed (5 ep)",
                )
        ax.set_title(dkey)
    fig.suptitle("Per-episode return by target drift (from train scenario of each step)")
    fig.text(
        0.5,
        0.02,
        "Instruction scenarios improved most; language scenarios are sparse in this run.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "per_drift_type_curve.png", dpi=120)
    plt.close(fig)


def _bar_from_eval_pair(
    base_path: Path, train_path: Path, title: str, caption: str, out: Path, order: list[str] | None = None
) -> None:
    if not base_path.is_file() or not train_path.is_file():
        print(f"skip bar: {base_path} or {train_path} missing", file=sys.stderr)
        return
    with open(base_path, encoding="utf-8") as f:
        bj = json.load(f)
    with open(train_path, encoding="utf-8") as f:
        tj = json.load(f)
    bbd = bj.get("aggregate", {}).get("by_drift_type", {})
    tbd = tj.get("aggregate", {}).get("by_drift_type", {})
    keys = order or sorted(set(bbd) | set(tbd))
    x = np.arange(len(keys))
    w = 0.35
    bmeans = [bbd.get(k, {}).get("mean", 0) for k in keys]
    bl = [bbd.get(k, {}).get("ci_lower", 0) for k in keys]
    bu = [bbd.get(k, {}).get("ci_upper", 0) for k in keys]
    berr = [abs(bmeans[i] - bl[i]) for i in range(len(keys))], [abs(bu[i] - bmeans[i]) for i in range(len(keys))]
    tmeans = [tbd.get(k, {}).get("mean", 0) for k in keys]
    tl = [tbd.get(k, {}).get("ci_lower", 0) for k in keys]
    tu = [tbd.get(k, {}).get("ci_upper", 0) for k in keys]
    terr = [abs(tmeans[i] - tl[i]) for i in range(len(keys))], [abs(tu[i] - tmeans[i]) for i in range(len(keys))]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x - w / 2,
        bmeans,
        w,
        yerr=berr,
        capsize=3,
        label="baseline (prompted Qwen2.5-1.5B)",
    )
    ax.bar(
        x + w / 2,
        tmeans,
        w,
        yerr=terr,
        capsize=3,
        label="trained (GRPO)",
    )
    for i, k in enumerate(keys):
        ax.text(
            i - w / 2,
            bmeans[i],
            f"{bmeans[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            i + w / 2,
            tmeans[i],
            f"{tmeans[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylabel("mean total reward (bootstrap CI err bars)")
    ax.set_title(title)
    ax.legend()
    if caption:
        fig.text(0.5, 0.01, caption, ha="center", fontsize=8)
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(REPO / "plots"),
        help="Output directory for PNGs",
    )
    args = ap.parse_args()
    out = Path(args.out)
    ev = Path(DATA_EVAL)
    plot_reward_curve(out)
    plot_per_drift_curve(out)
    _bar_from_eval_pair(
        ev / "in_domain_baseline.json",
        ev / "in_domain_trained.json",
        "In-domain eval: trained vs prompted baseline (5 seeds × 10 scenarios)",
        "Bootstrap 95% CIs; scenarios from data/scenarios/eval.jsonl (held out from training).",
        out / "eval_comparison.png",
    )
    _bar_from_eval_pair(
        ev / "transfer_baseline.json",
        ev / "transfer_trained.json",
        "Transfer: DearConnect (held out domain)",
        "Transfer to held-out DearConnect domain (not seen in training).",
        out / "transfer.png",
    )
    print(f"Wrote plots under {out}")


if __name__ == "__main__":
    main()
