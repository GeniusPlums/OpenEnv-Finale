#!/usr/bin/env bash
# V10 eval: baseline vs trained, in-domain + transfer, Hub uploads, FM-1..7 defenses.
# Intended for HF Jobs (a100-small). Run from repo root or let script clone to /workspace/role-drift-env.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

echo "V10_ENTRY ROLE_DRIFT_PERSONA_OPENAI_BASE_URL=${ROLE_DRIFT_PERSONA_OPENAI_BASE_URL-}"

# --- Resolve repository root -------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0})" && pwd)"
if [[ -f "$SCRIPT_DIR/../role_drift_env/__init__.py" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  cd "$REPO_ROOT"
elif [[ -d /workspace/role-drift-env/role_drift_env ]]; then
  cd /workspace/role-drift-env
else
  mkdir -p /workspace
  git clone --depth 1 "https://github.com/GeniusPlums/OpenEnv-Finale.git" /workspace/role-drift-env
  cd /workspace/role-drift-env
fi
export PYTHONPATH="$(pwd)"
echo "===== V10 Eval Job started at $(date) in $(pwd) ====="

# === FM-4: auth (fail before spend) =========================================
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "FATAL: HF_TOKEN unset. Aborting before spend."
  exit 1
fi
TOKEN="${HF_TOKEN:-$HUGGINGFACE_HUB_TOKEN}"
huggingface-cli login --token "$TOKEN" --add-to-git-credential 2>/dev/null || true

TRAINED_MODEL_REPO="GeniusPlums/role-drift-qwen-1-5b-grpo"
EVAL_RESULTS_REPO="GeniusPlums/role-drift-eval-results"

# === FM-2: trap + partial upload ============================================
upload_partial_results() {
  set +e
  echo "===== [trap] Uploading partial results to $EVAL_RESULTS_REPO ====="
  huggingface-cli repo create "$EVAL_RESULTS_REPO" --type dataset --private 2>/dev/null || true
  if [[ -d "data/eval_results" ]] && [[ -n "$(ls -A data/eval_results 2>/dev/null)" ]]; then
    huggingface-cli upload "$EVAL_RESULTS_REPO" "data/eval_results/" . \
      --repo-type dataset \
      --commit-message "Eval results (V10, partial-or-final)" || echo "WARN: upload failed"
  else
    echo "WARN: no eval results to upload"
  fi
  if [[ -n "${NVIDIA_MONITOR_PID:-}" ]]; then kill "$NVIDIA_MONITOR_PID" 2>/dev/null || true; fi
  if [[ -n "${VLLM_PID:-}" ]]; then kill "$VLLM_PID" 2>/dev/null || true; fi
  set -e
}
trap upload_partial_results EXIT

# FM-3: nvidia-smi every 30 minutes — stdout only (job log stream), not /tmp (lost on teardown)
(
  while true; do
    echo "===== nvidia-smi $(date) ====="
    nvidia-smi || true
    sleep 1800
  done
) &
NVIDIA_MONITOR_PID=$!

mkdir -p data/eval_results

pip install -e . --quiet
pip install -q vllm sentence-transformers langdetect peft accelerate huggingface_hub openai

# === Download trained checkpoint ==============================================
echo "===== Downloading trained checkpoint ====="
huggingface-cli download "$TRAINED_MODEL_REPO" \
  --local-dir checkpoints/trained \
  --include "*.json" --include "*.safetensors" --include "*.model" --include "*.txt" || true

# === FM-3: nvidia-smi before model loads ======================================
nvidia-smi

# === Customer-sim vLLM ========================================================
VLLM_PORT="${VLLM_PORT:-8000}"
export ROLE_DRIFT_PERSONA_OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
echo "[V10] ROLE_DRIFT_PERSONA_OPENAI_BASE_URL=$ROLE_DRIFT_PERSONA_OPENAI_BASE_URL"

echo "===== Starting customer-sim vLLM ====="
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port "$VLLM_PORT" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.40 \
  > vllm_server.log 2>&1 &
VLLM_PID=$!

VLLM_OK=0
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
    echo "vLLM ready after ~$((i * 5))s"
    VLLM_OK=1
    break
  fi
  sleep 5
done
if [[ "$VLLM_OK" -ne 1 ]]; then
  echo "FATAL: vLLM not ready in 5 min (see vllm_server.log)"
  tail -120 vllm_server.log 2>/dev/null || true
  exit 1
fi

# === FM-1: persona gate (no silent fallback) =================================
if ! python -c "
import os, sys
assert os.environ.get('ROLE_DRIFT_PERSONA_OPENAI_BASE_URL')
from role_drift_env.server.environment import RoleDriftEnvironment
from role_drift_env.server.personas import load_llm_persona
env = RoleDriftEnvironment()
obs, state = env.reset('coop_kk_06')
state.history = [{'role': 'agent', 'text': 'Thanks for your time. Goodbye.'}]
p = load_llm_persona('thank_you_bomber')
out = p.next_utterance(state, 42)
print('PERSONA_OUTPUT:', (out or '')[:300])
FALL = 'Thanks, I think I have what I need. Goodbye.'
is_fb = (FALL in (out or '')) or (len((out or '').strip()) < 2)
print('IS_FALLBACK:', is_fb)
sys.exit(1 if is_fb else 0)
"; then
  echo "FATAL: customer-sim returned fallback. Aborting."
  exit 1
fi
echo "===== Persona check PASSED ====="
# FM-7: do not touch role_drift_env/server/rewards/ for this bundle.

# === Eval runs (one policy per process: bash invokes python multiple times) =
run_eval_block() {
  local label="$1"
  shift
  echo "===== Eval: $label ====="
  python "$@"
  nvidia-smi
  python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"
}

# 1. In-domain BASELINE
run_eval_block "in_domain BASELINE" scripts/run_eval.py in_domain \
  --policy-checkpoint Qwen/Qwen2.5-1.5B-Instruct \
  --scenarios-jsonl data/scenarios/eval.jsonl \
  --seeds 0,1,2,3,4 \
  --label baseline_qwen_1_5b \
  --output-json data/eval_results/in_domain_baseline.json \
  --upload-on-each-scenario "$EVAL_RESULTS_REPO"

# 2. In-domain TRAINED
run_eval_block "in_domain TRAINED" scripts/run_eval.py in_domain \
  --policy-checkpoint checkpoints/trained \
  --scenarios-jsonl data/scenarios/eval.jsonl \
  --seeds 0,1,2,3,4 \
  --label trained \
  --output-json data/eval_results/in_domain_trained.json \
  --upload-on-each-scenario "$EVAL_RESULTS_REPO"

# 3. Transfer BASELINE
run_eval_block "transfer BASELINE" scripts/run_eval.py transfer \
  --policy-checkpoint Qwen/Qwen2.5-1.5B-Instruct \
  --scenarios-jsonl data/scenarios/transfer_dearconnect.jsonl \
  --seeds 0,1,2,3,4 \
  --label baseline_qwen_1_5b \
  --output-json data/eval_results/transfer_baseline.json \
  --upload-on-each-scenario "$EVAL_RESULTS_REPO"

# 4. Transfer TRAINED
run_eval_block "transfer TRAINED" scripts/run_eval.py transfer \
  --policy-checkpoint checkpoints/trained \
  --scenarios-jsonl data/scenarios/transfer_dearconnect.jsonl \
  --seeds 0,1,2,3,4 \
  --label trained \
  --output-json data/eval_results/transfer_trained.json \
  --upload-on-each-scenario "$EVAL_RESULTS_REPO"

# 5. Reward-hacking probes
echo "===== Reward-hacking probes ====="
python scripts/reward_hacking_probes.py \
  --scenarios-jsonl data/scenarios/eval.jsonl \
  --output-json data/eval_results/reward_hacking_probes.json

# Final upload
echo "===== Final upload ====="
set +e
huggingface-cli upload "$EVAL_RESULTS_REPO" data/eval_results/ . \
  --repo-type dataset \
  --commit-message "Eval results V10 (final)"
set -e

echo "===== Verifying upload ====="
rm -rf /tmp/v10_verify
mkdir -p /tmp/v10_verify
huggingface-cli download "$EVAL_RESULTS_REPO" --include "in_domain_trained.json" \
  --local-dir /tmp/v10_verify --repo-type dataset
ls -la /tmp/v10_verify/

echo "===== V10 Eval complete at $(date) ====="
