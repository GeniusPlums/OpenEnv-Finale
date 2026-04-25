#!/bin/bash
# Full evaluation script - run when compute credits arrive

set -e

echo "=== Role Drift Env - Full Evaluation ==="

# Training (if not already done)
if [ -z "$1" ]; then
    echo "Note: Training must be run first with: python training/train_grpo.py --episodes 150"
    echo "Skipping training..."
fi

# Run evals in parallel
echo "Running in-domain eval..."
python training/eval_baseline.py --model-path checkpoints/grpo_final/best --scenario-file data/scenarios/eval.jsonl --num-seeds 5 --output data/eval_results/trained_eval.jsonl

echo "Running transfer eval (DearConnect)..."
python training/eval_baseline.py --model-path checkpoints/grpo_final/best --scenario-file data/scenarios/transfer_dearconnect.jsonl --num-seeds 5 --output data/eval_results/trained_transfer.jsonl

echo "Running held-out persona eval..."
python training/eval_baseline.py --model-path checkpoints/grpo_final/best --scenario-file data/scenarios/eval_held_out_persona.jsonl --num-seeds 5 --output data/eval_results/trained_held_out.jsonl

echo "Running prompt injection eval..."
python training/eval_baseline.py --model-path checkpoints/grpo_final/best --scenario-file data/scenarios/eval_injection.jsonl --num-seeds 5 --output data/eval_results/trained_injection.jsonl

# Run reward hacking probes
echo "Running reward hacking probes..."
python scripts/reward_hacking_probes.py --checkpoint checkpoints/grpo_final/best

# Generate plots
echo "Generating plots..."
python scripts/plot_training.py data/training_logs/run_final/episode_log.jsonl --out plots/run_final/

python scripts/regenerate_baseline_vs_trained_plot.py

echo "=== Evaluation complete ==="
echo "Results in data/eval_results/"
echo "Plots in plots/"