#!/usr/bin/env bash
# V10 eval: baseline vs trained, in-domain + transfer, Hub uploads, FM-1..7 defenses.
# Intended for HF Jobs (a100-small). Run from repo root or let script clone to /workspace/role-drift-env.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
# HF Jobs H200: align with V9 — avoid premature NVML CUDA init before fabric is ready.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

echo "V10_ENTRY ROLE_DRIFT_PERSONA_OPENAI_BASE_URL=${ROLE_DRIFT_PERSONA_OPENAI_BASE_URL-}"

# --- Resolve repository root -------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
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

TRAINED_MODEL_REPO="GeniusPlums/role-drift-qwen-1-5b-grpo"
# Optional: pin a Hub git revision (commit SHA or branch) for reproducible eval.
TRAINED_MODEL_REVISION="${TRAINED_MODEL_REVISION:-}"
EVAL_RESULTS_REPO="GeniusPlums/role-drift-eval-results"

# === FM-2: trap + partial upload ============================================
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

# CUDA torch first (HF python:3.11 image often resolves CPU-only torch from `pip install -e .`).
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e . --quiet
pip install -q vllm sentence-transformers langdetect peft accelerate huggingface_hub openai
# `hf` is from huggingface_hub; login before Hub download/upload
hf auth login --token "$TOKEN" --add-to-git-credential

# === Download trained checkpoint ==============================================
# `huggingface-cli` is broken in many images; `hf download` is required
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
  echo "FATAL: no checkpoints/trained/config.json after hf download. Aborting."
  exit 1
fi

# === FM-3: nvidia-smi before any PyTorch CUDA calls ============================
echo "===== nvidia-smi (before fabric wait) ====="
nvidia-smi

# GPU fabric on HF can lag nvidia-smi; wait before first torch CUDA probe (Error 802).
_FABRIC_SLEEP="${CUDA_FABRIC_SLEEP_SEC:-100}"
echo "===== Post-nvidia-smi fabric settle (${_FABRIC_SLEEP}s, override CUDA_FABRIC_SLEEP_SEC; use 90–120 on H200) ====="
sleep "$_FABRIC_SLEEP"

# Subprocess per attempt — same as V9 training; no torch in parent yet for probe.
export CUDA_WAIT_PRE_PROBE_SEC="${CUDA_WAIT_PRE_PROBE_SEC:-30}"
echo "===== Active CUDA wait (scripts/wait_for_cuda.py, CUDA_WAIT_*) ====="
python scripts/wait_for_cuda.py || {
  echo "FATAL: CUDA not ready after wait_for_cuda (802 / driver race). Increase CUDA_FABRIC_SLEEP_SEC or CUDA_WAIT_PRE_PROBE_SEC."
  exit 1
}

# Warmup in THIS process so vLLM inherits a known-good CUDA context state.
echo "===== CUDA warmup (before vLLM; confirms GPU ready) ====="
python -c "
import torch
print('[V10] torch', torch.__version__, 'cuda', torch.version.cuda, flush=True)
avail = torch.cuda.is_available()
print('[V10] cuda.is_available() =', avail, flush=True)
if not avail:
    raise SystemExit('CUDA not available after wait_for_cuda')
torch.cuda.set_device(0)
_ = torch.cuda.current_device()
n = torch.cuda.device_count()
print('[V10] cuda.device_count() =', n, flush=True)
x = torch.zeros(2, device='cuda', dtype=torch.float32)
torch.cuda.synchronize()
del x
print('[V10] GPU ready:', torch.cuda.get_device_name(0), flush=True)
"

# === Customer-sim vLLM (retries for engine init / 802 races) ==================
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
      echo "[V10] sleeping ${VLLM_RETRY_SLEEP_SEC}s before vLLM retry (CUDA_WAIT / fabric)..."
      sleep "$VLLM_RETRY_SLEEP_SEC"
      echo "[V10] post-retry CUDA ping:"
      python -c "import torch; torch.cuda.synchronize(); print('[V10] device_count', torch.cuda.device_count())" || true
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
    echo "[V10] vLLM attempt ${attempt} failed (no HTTP /models in 5 min or process exit)"
    tail -100 vllm_server.log 2>/dev/null || true
  done
  return 1
}

if ! start_vllm_with_retries; then
  echo "FATAL: vLLM failed after ${VLLM_MAX_START_ATTEMPTS} attempts"
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
