#!/usr/bin/env bash
# V8 closeout: run on the HF GPU runtime where training completed.
# See OPENCODE_BRIEF_V8_RUNTIME.md. Order: verify → Hub → sanity → baselines (if needed) →
# trained in-domain → print headline → trained transfer → print headline → reward probe → plots → verify Hub
set -euo pipefail

export HF_USER="${HF_USER:-GeniusPlums}"
export REPO_ROOT="${REPO_ROOT:-$(pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

HUB_DIR="${HUB_DIR:-/tmp/hub_verify}"
SANITY_JSON="${SANITY_JSON:-/tmp/sanity_rollout.json}"

# --- Step 1: Verifying artifacts (brief: PASS / REFUSED) ---
echo "Step 1: Verifying artifacts..."
if [[ ! -d checkpoints/grpo_final/best ]] || [[ ! -d checkpoints/grpo_final/final ]]; then
  echo "Step 1: REFUSED — missing checkpoints/grpo_final/best or final"
  exit 1
fi
LOG="data/training_logs/run_final/episode_log.jsonl"
if [[ ! -f "$LOG" ]]; then
  echo "Step 1: REFUSED — missing $LOG"
  exit 1
fi
N=$(wc -l < "$LOG" | tr -d ' ')
if [[ "$N" -ne 100 ]]; then
  echo "Step 1: REFUSED — expected 100 lines in episode_log.jsonl, got $N"
  exit 1
fi
ls -lh checkpoints/grpo_final/best/
ls -lh checkpoints/grpo_final/final/
du -sh checkpoints/grpo_final/best/ || true
head -1 "$LOG" | python -c "import sys, json; d=json.loads(sys.stdin.read()); print('keys:', sorted(d.keys())); print('component_means sample:', d.get('component_means'))"
tail -1 "$LOG" | python -c "import sys, json; d=json.loads(sys.stdin.read()); print('final episode index:', d.get('episode'))"
echo "Step 1: PASS"
echo

# --- Step 2: Hub upload ---
echo "Step 2: Pushing to Hub (best checkpoint)..."
huggingface-cli whoami
hub_push() {
  huggingface-cli upload "${HF_USER}/role-drift-qwen-1-5b-grpo" \
    checkpoints/grpo_final/best \
    --repo-type model \
    --commit-message "GRPO-trained Qwen2.5-1.5B on role-drift environment, 100 episodes"
}
hub_push || (echo "Retrying Hub upload once..." && hub_push)

rm -rf "$HUB_DIR"
huggingface-cli download "${HF_USER}/role-drift-qwen-1-5b-grpo" --include="config.json" --quiet --local-dir "$HUB_DIR"
ls -l "$HUB_DIR"
echo "Hub upload verified (config.json in $HUB_DIR)."
echo

# --- Step 3: Sanity rollout ---
echo "Step 3: Running sanity rollout (one episode)..."
python scripts/sanity_trained_one_episode.py \
  --model-path checkpoints/grpo_final/best \
  --scenario-file data/scenarios/eval.jsonl \
  --output "$SANITY_JSON"
python -c "
import json, sys
d = json.load(open('$SANITY_JSON'))
r = d.get('episode_return')
s = d.get('first_agent_utterance') or ''
print('Sanity episode_return:', r, '(must be finite)')
print('first_agent_utterance (first 300 chars):', s[:300])
if r is None or (isinstance(r, float) and (r != r)):
    print('REFUSE: bad episode_return', file=sys.stderr)
    sys.exit(1)
if not s.strip():
    print('REFUSE: empty first agent utterance', file=sys.stderr)
    sys.exit(1)
"
echo "Step 3: PASS (non-empty utterance, finite return)"
echo

# --- Step 4: Baselines first (so headlines have baseline ready), then trained evals ---
echo "Step 4: Ensuring prompted baselines (Qwen2.5-1.5B-Instruct) if missing..."
if [[ ! -f data/eval_results/baseline_qwen_1_5b.json ]]; then
  echo "  Running in-domain baseline (this is slow)..."
  python training/eval_baseline.py \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --scenario-file data/scenarios/eval.jsonl \
    --num-seeds 5 \
    --output data/eval_results/baseline_qwen_1_5b.json
fi
if [[ ! -f data/eval_results/baseline_qwen_1_5b_transfer.json ]]; then
  echo "  Running transfer baseline (this is slow)..."
  python training/eval_baseline.py \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --scenario-file data/scenarios/transfer_dearconnect.jsonl \
    --num-seeds 5 \
    --output data/eval_results/baseline_qwen_1_5b_transfer.json
fi

echo "Step 4: Running in-domain eval (trained checkpoint) — ~15–20 min"
python training/eval_baseline.py \
  --model-path checkpoints/grpo_final/best \
  --scenario-file data/scenarios/eval.jsonl \
  --num-seeds 5 \
  --output data/eval_results/trained_qwen_1_5b.json

echo
echo ">>> IN-DOMAIN HEADLINE (copy for Palak) <<<"
python scripts/print_closeout_headlines.py --in-domain
echo

echo "Step 4: Running transfer eval (trained checkpoint) — ~15–20 min"
python training/eval_baseline.py \
  --model-path checkpoints/grpo_final/best \
  --scenario-file data/scenarios/transfer_dearconnect.jsonl \
  --num-seeds 5 \
  --output data/eval_results/trained_qwen_1_5b_transfer.json

echo
echo ">>> TRANSFER HEADLINE (copy for Palak) <<<"
python scripts/print_closeout_headlines.py --transfer
echo

echo "Step 4: Running reward-hacking probe (trivial + trained)..."
python scripts/reward_hacking_probes.py \
  --checkpoint checkpoints/grpo_final/best \
  --output data/eval_results/reward_hacking_probe_complete.json \
  --plot

echo "Step 4: Baseline summary (prompted) for the log"
python -c "
import json
for path, label in [
    ('data/eval_results/baseline_qwen_1_5b.json', 'IN-DOMAIN BASELINE'),
    ('data/eval_results/baseline_qwen_1_5b_transfer.json', 'TRANSFER BASELINE'),
]:
    d = json.load(open(path))
    s = d.get('summary', {})
    print(label, 'mean_return=', s.get('mean_return'), 'n_episodes=', s.get('num_episodes'))
"
echo

# --- Step 5: Plots (brief Step 6 / 7 naming) ---
echo "Step 5: Generating plots"
python scripts/plot_training.py data/training_logs/run_final/episode_log.jsonl --out plots/run_final/
python scripts/regenerate_baseline_vs_trained_plot.py
python scripts/regenerate_in_domain_vs_transfer_plot.py
echo

# --- Re-verify Hub ---
echo "Re-verifying Hub model repo..."
rm -rf "$HUB_DIR"
huggingface-cli download "${HF_USER}/role-drift-qwen-1-5b-grpo" --include="config.json" --quiet --local-dir "$HUB_DIR"
ls -l "$HUB_DIR"

# --- List outputs (Step 4 verify in V8 brief) ---
echo
echo "Output inventory:"
ls -lh data/eval_results/ 2>/dev/null || true
ls -lh plots/baseline_vs_trained.png plots/in_domain_vs_transfer.png plots/reward_hacking_probe.png 2>/dev/null || true
ls -d plots/run_final/ 2>/dev/null || true
ls -lh plots/run_final/ 2>/dev/null | head -20 || true

echo
echo "=============================================================="
echo "CLOSEOUT COMPLETE."
echo
echo "The following are now safe and persisted (checkpoint on Hub; eval+plots on disk):"
echo "  - HF Hub: ${HF_USER}/role-drift-qwen-1-5b-grpo"
echo "  - data/eval_results/*.json"
echo "  - plots/ (incl. baseline_vs_trained, in_domain_vs_transfer, reward_hacking, run_final/)"
echo
echo "NEXT ACTION FOR YOU — STOP the runtime in the Hugging Face UI (Endpoints / GPU)."
echo "Idle billing is about \$0.07/min. Do not only close the browser tab."
echo
echo "After teardown: local git commit (data/eval_results, plots, data/training_logs, closeout.log),"
echo "  docs/hypotheses.md, README, BENCHMARK.md, send Palak the headline plots + numbers."
echo "=============================================================="

echo
echo "Optional: git add data/eval_results/ plots/ data/training_logs/run_final/ closeout.log 2>/dev/null; git commit and push if auth works on this box."
