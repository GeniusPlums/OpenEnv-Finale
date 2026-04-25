"""
Print in-domain and transfer tables from eval_baseline JSONs (summary.mean_return, per_detector_mean).
Run on the training runtime right after each trained eval file is written.
"""
import argparse
import json
import os
import sys


def print_in_domain() -> bool:
    bp = "data/eval_results/baseline_qwen_1_5b.json"
    tp = "data/eval_results/trained_qwen_1_5b.json"
    if not os.path.isfile(bp) or not os.path.isfile(tp):
        return False
    base = json.load(open(bp, encoding="utf-8"))["summary"]
    trn = json.load(open(tp, encoding="utf-8"))["summary"]

    print("=" * 60)
    print("IN-DOMAIN HEADLINE NUMBERS (eval_baseline.py schema)")
    print("=" * 60)
    print(f"{'':20} {'Baseline':>12} {'Trained':>12} {'Delta':>12}")
    print("-" * 60)

    def row(label: str, b: float, t: float) -> None:
        d = t - b
        sign = "+" if d >= 0 else ""
        print(f"{label:<20} {b:>12.4f} {t:>12.4f} {sign}{d:>11.4f}")

    b_mean = base.get("mean_return")
    t_mean = trn.get("mean_return")
    if b_mean is not None and t_mean is not None:
        row("Aggregate (return)", b_mean, t_mean)
    bpd = base.get("per_detector_mean") or {}
    tpd = trn.get("per_detector_mean") or {}
    for k, label in [
        ("term", "Termination"),
        ("goal", "Goal"),
        ("instr", "Instruction"),
        ("lang", "Language"),
    ]:
        if k in bpd or k in tpd:
            row(label, bpd.get(k, 0.0), tpd.get(k, 0.0))

    print()
    print(f"Baseline n_episodes: {base.get('n_episodes', '?')}")
    print(f"Trained n_episodes:  {trn.get('n_episodes', '?')}")
    print("=" * 60)
    return True


def print_transfer() -> bool:
    bp = "data/eval_results/baseline_qwen_1_5b_transfer.json"
    tp = "data/eval_results/trained_qwen_1_5b_transfer.json"
    if not os.path.isfile(bp) or not os.path.isfile(tp):
        return False
    b = json.load(open(bp, encoding="utf-8"))["summary"]
    t = json.load(open(tp, encoding="utf-8"))["summary"]
    print("=" * 60)
    print("TRANSFER (DearConnect) — mean episode return")
    print("=" * 60)
    print(f"Baseline: {b.get('mean_return')}")
    print(f"Trained:  {t.get('mean_return')}")
    d = t.get("mean_return", 0) - b.get("mean_return", 0)
    sign = "+" if d >= 0 else ""
    print(f"Delta:    {sign}{d:.4f}")
    print(f"Baseline n_episodes: {b.get('n_episodes', '?')}")
    print(f"Trained  n_episodes: {t.get('n_episodes', '?')}")
    print("=" * 60)
    return True


def print_full_summaries() -> None:
    for path, name in [
        ("data/eval_results/baseline_qwen_1_5b.json", "BASELINE IN-DOMAIN"),
        ("data/eval_results/trained_qwen_1_5b.json", "TRAINED IN-DOMAIN"),
    ]:
        if os.path.isfile(path):
            d = json.load(open(path, encoding="utf-8"))
            print(f"=== {name} — summary ===")
            print(json.dumps(d.get("summary", {}), indent=2))
            print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-domain", action="store_true", help="Only in-domain table")
    ap.add_argument("--transfer", action="store_true", help="Only transfer table")
    ap.add_argument("--dump", action="store_true", help="Dump full summary JSON blocks")
    args = ap.parse_args()
    if args.dump:
        print_full_summaries()
    elif args.in_domain:
        if not print_in_domain():
            sys.exit(1)
    elif args.transfer:
        if not print_transfer():
            sys.exit(1)
    else:
        print_in_domain()
        print()
        print_transfer()
