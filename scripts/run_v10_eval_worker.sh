#!/usr/bin/env bash
# V10 eval worker: one full attempt (bootstrap, CUDA gate, vLLM, eval). Invoked by run_v10_eval_job.sh.
# Exits non-zero on CUDA/vLLM failure so the wrapper can restart a clean bash process.
#
# shellcheck disable=SC1091
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

REPO_ROOT="${V10_REPO_ROOT:-}"
if [[ -z "$REPO_ROOT" ]] || [[ ! -f "$REPO_ROOT/role_drift_env/__init__.py" ]]; then
  echo "FATAL: V10_REPO_ROOT must point at repo root (role_drift_env present). Got: ${REPO_ROOT:-empty}"
  exit 1
fi
cd "$REPO_ROOT"
export PYTHONPATH="$(pwd)"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "FATAL: HF_TOKEN unset."
  exit 1
fi
TOKEN="${HF_TOKEN:-$HUGGINGFACE_HUB_TOKEN}"

echo "===== V10 worker attempt ${V10_EVAL_RESTART:-1} at $(date) in $(pwd) ====="

TRAINED_MODEL_REPO="GeniusPlums/role-drift-qwen-1-5b-grpo"
TRAINED_MODEL_REVISION="${TRAINED_MODEL_REVISION:-}"
EVAL_RESULTS_REPO="GeniusPlums/role-drift-eval-results"

upload_partial_results() {
  set +e
  echo "===== [trap] Uploading partial results to $EVAL_RESULTS_REPO ====="
  if [[ -d "data/eval_results" ]] && [[ -n "$(ls -A data/eval_results 2>/dev/null)" ]]; then
    hf upload "$EVAL_RESULTS_REPO" "data/eval_results" . \
      --repo-type dataset \
      --private \
      --commit-message "Eval results (V10, partial-or-final)" || echo "WARN: upload failed"
  else
    echo "WARN: no eval results to upload"
  fi
  if [[ -n "${NVIDIA_MONITOR_PID:-}" ]]; then kill "$NVIDIA_MONITOR_PID" 2>/dev/null || true; fi
  if [[ -n "${VLLM_PID:-}" ]]; then kill "$VLLM_PID" 2>/dev/null || true; fi
  set -e
}
trap upload_partial_results EXIT

(
  while true; do
    echo "===== nvidia-smi $(date) ====="
    nvidia-smi || true
    sleep 1800
  done
) &
NVIDIA_MONITOR_PID=$!

mkdir -p data/eval_results

# Bootstrap pip only on first attempt (or if forced) — retries get a new bash but reuse venv/site-packages.
_RESTART_NUM="${V10_EVAL_RESTART:-1}"
if [[ "${V10_ALWAYS_PIP:-0}" == "1" ]] || [[ "$_RESTART_NUM" -eq 1 ]]; then
  echo "===== pip install (restart ${_RESTART_NUM}) ====="
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu126
  pip install -e . --quiet
  pip install -q vllm sentence-transformers langdetect peft accelerate huggingface_hub openai
else
  echo "===== pip skipped (V10_EVAL_RESTART=${_RESTART_NUM}; V10_ALWAYS_PIP=1 to reinstall) ====="
fi
hf auth login --token "$TOKEN" --add-to-git-credential

echo "===== Downloading trained checkpoint ====="
mkdir -p checkpoints/trained
HF_DL_ARGS=("$TRAINED_MODEL_REPO" --local-dir checkpoints/trained \
  --include "*.json" --include "*.safetensors" --include "*.model" --include "*.txt")
if [[ -n "$TRAINED_MODEL_REVISION" ]]; then
  HF_DL_ARGS+=(--revision "$TRAINED_MODEL_REVISION")
  echo "[V10] Using TRAINED_MODEL_REVISION=$TRAINED_MODEL_REVISION"
fi
hf download "${HF_DL_ARGS[@]}"
if [[ ! -f "checkpoints/trained/config.json" ]]; then
  echo "FATAL: no checkpoints/trained/config.json after hf download."
  exit 1
fi

# --- CUDA path: no torch imports above this line --------------------------------
echo "===== nvidia-smi (before fabric wait) ====="
nvidia-smi

_FABRIC_SLEEP="${CUDA_FABRIC_SLEEP_SEC:-105}"
echo "===== Fabric settle (${_FABRIC_SLEEP}s, CUDA_FABRIC_SLEEP_SEC) ====="
sleep "$_FABRIC_SLEEP"

# Fail fast: ~12–15 probes × 5s ≈ 60–75s, not 5+ minutes.
export CUDA_WAIT_MAX_RETRIES="${CUDA_WAIT_MAX_RETRIES:-12}"
export CUDA_WAIT_SLEEP_SEC="${CUDA_WAIT_SLEEP_SEC:-5}"
export CUDA_WAIT_PRE_PROBE_SEC="${CUDA_WAIT_PRE_PROBE_SEC:-30}"
echo "===== wait_for_cuda.py (max ${CUDA_WAIT_MAX_RETRIES} tries × ${CUDA_WAIT_SLEEP_SEC}s; unrecoverable if exhausted) ====="
if ! python scripts/wait_for_cuda.py; then
  echo "FATAL [exit 2]: CUDA unrecoverable after ${CUDA_WAIT_MAX_RETRIES} subprocess probes (~$((CUDA_WAIT_MAX_RETRIES * CUDA_WAIT_SLEEP_SEC))s) + pre-probe sleep."
  echo "FATAL: This container GPU/driver may be broken (802). Do not wait longer — retry full job or new HF allocation."
  exit 2
fi

echo "===== CUDA sanity (fail immediately if any call errors) ====="
python -c "
import sys
import torch
print('[V10] torch', torch.__version__, 'cuda', torch.version.cuda, flush=True)
if not torch.cuda.is_available():
    print('FATAL: torch.cuda.is_available() is False', flush=True)
    sys.exit(3)
torch.cuda.set_device(0)
try:
    n = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
except Exception as e:
    print('FATAL: device_count/get_device_name failed:', e, flush=True)
    sys.exit(3)
print('[V10] device_count=', n, 'name=', name, flush=True)
if n < 1:
    sys.exit(3)
x = torch.zeros(2, device='cuda', dtype=torch.float32)
torch.cuda.synchronize()
del x
print('[V10] CUDA sanity OK', flush=True)
" || { echo "FATAL [exit 3]: CUDA sanity check failed"; exit 3; }

VLLM_PORT="${VLLM_PORT:-8000}"
export ROLE_DRIFT_PERSONA_OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
echo "[V10] ROLE_DRIFT_PERSONA_OPENAI_BASE_URL=$ROLE_DRIFT_PERSONA_OPENAI_BASE_URL"

VLLM_MAX_START_ATTEMPTS="${VLLM_MAX_START_ATTEMPTS:-3}"
VLLM_RETRY_SLEEP_SEC="${VLLM_RETRY_SLEEP_SEC:-25}"

start_vllm_with_retries() {
  local attempt ok i
  VLLM_PID=""
  for attempt in $(seq 1 "$VLLM_MAX_START_ATTEMPTS"); do
    echo "===== vLLM start attempt ${attempt}/${VLLM_MAX_START_ATTEMPTS} ====="
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "[V10] stopping previous vLLM pid=$VLLM_PID"
      kill "$VLLM_PID" 2>/dev/null || true
      wait "$VLLM_PID" 2>/dev/null || true
    fi
    VLLM_PID=""
    if [[ "$attempt" -gt 1 ]]; then
      echo "[V10] sleeping ${VLLM_RETRY_SLEEP_SEC}s before vLLM retry..."
      sleep "$VLLM_RETRY_SLEEP_SEC"
    fi
    echo "[V10] launching vLLM (enforce-eager; log: vllm_server.log)..."
    python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-7B-Instruct \
      --port "$VLLM_PORT" \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.40 \
      --enforce-eager \
      > vllm_server.log 2>&1 &
    VLLM_PID=$!
    echo "[V10] vLLM PID=$VLLM_PID"

    ok=0
    for i in $(seq 1 60); do
      if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
        echo "[V10] vLLM HTTP ready after ~$((i * 5))s (attempt ${attempt})"
        ok=1
        break
      fi
      if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[V10] vLLM process died during startup (attempt ${attempt}). Log tail:"
        tail -100 vllm_server.log 2>/dev/null || true
        break
      fi
      sleep 5
    done
    if [[ "$ok" -eq 1 ]]; then
      return 0
    fi
    echo "[V10] vLLM attempt ${attempt} failed"
    tail -100 vllm_server.log 2>/dev/null || true
  done
  return 1
}

if ! start_vllm_with_retries; then
  echo "FATAL: vLLM failed after ${VLLM_MAX_START_ATTEMPTS} attempts"
  exit 1
fi

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

run_eval_block() {
  local label="$1"
  shift
  echo "===== Eval: $label ====="
  python "$@"
  nvidia-smi
  python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"
}

run_eval_block "in_domain BASELINE" scripts/run_eval.py in_domain \
  --policy-checkpoint Qwen/Qwen2.5-1.5B-Instruct \
  --scenarios-jsonl data/scenarios/eval.jsonl \
  --seeds 0,1,2,3,4 \
  --label baseline_qwen_1_5b \
  --output-json data/eval_results/in_domain_baseline.json \
  --upload-on-each-scenario "$EVAL_RESULTS_REPO"

run_eval_block "in_domain TRAINED" scripts/run_eval.py in_domain \
  --policy-checkpoint checkpoints/trained \
  --scenarios-jsonl data/scenarios/eval.jsonl \
  --seeds 0,1,2,3,4 \
  --label trained \
  --output-json data/eval_results/in_domain_trained.json \
  --upload-on-each-scenario "$EVAL_RESULTS_REPO"

run_eval_block "transfer BASELINE" scripts/run_eval.py transfer \
  --policy-checkpoint Qwen/Qwen2.5-1.5B-Instruct \
  --scenarios-jsonl data/scenarios/transfer_dearconnect.jsonl \
  --seeds 0,1,2,3,4 \
  --label baseline_qwen_1_5b \
  --output-json data/eval_results/transfer_baseline.json \
  --upload-on-each-scenario "$EVAL_RESULTS_REPO"

run_eval_block "transfer TRAINED" scripts/run_eval.py transfer \
  --policy-checkpoint checkpoints/trained \
  --scenarios-jsonl data/scenarios/transfer_dearconnect.jsonl \
  --seeds 0,1,2,3,4 \
  --label trained \
  --output-json data/eval_results/transfer_trained.json \
  --upload-on-each-scenario "$EVAL_RESULTS_REPO"

echo "===== Reward-hacking probes ====="
python scripts/reward_hacking_probes.py \
  --scenarios-jsonl data/scenarios/eval.jsonl \
  --output-json data/eval_results/reward_hacking_probes.json

echo "===== Final upload ====="
set +e
hf upload "$EVAL_RESULTS_REPO" "data/eval_results" . \
  --repo-type dataset \
  --commit-message "Eval results V10 (final)"
set -e

echo "===== Verifying upload ====="
rm -rf /tmp/v10_verify
mkdir -p /tmp/v10_verify
hf download "$EVAL_RESULTS_REPO" "in_domain_trained.json" \
  --local-dir /tmp/v10_verify --repo-type dataset
ls -la /tmp/v10_verify/

echo "===== V10 Eval complete at $(date) ====="
trap - EXIT
exit 0
