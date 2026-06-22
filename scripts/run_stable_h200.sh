#!/usr/bin/env bash
# Stable H200 HF Jobs launcher: fabric settle + wait_for_cuda (802 avoidance), then vLLM, then GRPO.
# Precondition: this script must be the only application entry after image setup — first Python
# that touches CUDA ordering is scripts/wait_for_cuda.py (its probe children run inside that step).

if [ ! -z "$ALREADY_RUNNING" ]; then
  echo "Duplicate execution detected, exiting"
  exit 1
fi
export ALREADY_RUNNING=1

set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

# Do not set ALLOW_CUDA=1 before wait_for_cuda completes; no vLLM / training / torch before then.

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

export CUDA_FABRIC_SLEEP_SEC=100
export CUDA_WAIT_PRE_PROBE_SEC=60

VLLM_PORT="${VLLM_PORT:-8000}"
export ROLE_DRIFT_PERSONA_OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

export HF_USER="${HF_USER:-GeniusPlums}"
export TRAINED_REPO="${TRAINED_REPO:-GeniusPlums/role-drift-qwen-1-5b-grpo}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"

echo "===== nvidia-smi ====="
nvidia-smi

echo "===== CUDA fabric settle (${CUDA_FABRIC_SLEEP_SEC}s) ====="
sleep "${CUDA_FABRIC_SLEEP_SEC}"

echo "===== FIRST PYTHON (application): scripts/wait_for_cuda.py (CUDA_WAIT_PRE_PROBE_SEC=${CUDA_WAIT_PRE_PROBE_SEC}) ====="
python scripts/wait_for_cuda.py

# ONLY AFTER success:
export ROLE_DRIFT_SKIP_TRAIN_ENTRY_CUDA_WAIT=1
export ALLOW_CUDA=1

echo "===== DEBUG: confirm ALLOW_CUDA before vLLM (must be 1) ====="
python -c "import os, sys; v = os.environ.get('ALLOW_CUDA'); print('ALLOW_CUDA=', v, flush=True); sys.exit(0 if v == '1' else 1)"

echo "===== Starting vLLM (OpenAI API server) ====="
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port "$VLLM_PORT" \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.30 \
  --enforce-eager \
  > vllm_server.log 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

echo "===== Waiting for vLLM HTTP ====="
VLLM_OK=0
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
    echo "vLLM ready after $i attempt(s) (~$((i * 5))s)"
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

echo "===== GRPO training (train_grpo.py) ====="
python training/train_grpo.py \
  --episodes 100 \
  --group-size 4 \
  --lr 5e-6 \
  --kl-coef 0.125 \
  --lang-oversample 2 \
  --term-oversample 1 \
  --curriculum adversarial \
  --policy-model Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint-every 25 \
  --max-turns 6 \
  --output-dir data/training_logs/run_final \
  --checkpoint-dir checkpoints/grpo_final \
  --hub-repo "$TRAINED_REPO" \
  2>&1 | tee training.log

echo "===== run_stable_h200.sh finished at $(date) ====="
