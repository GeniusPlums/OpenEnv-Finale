#!/usr/bin/env bash
# V9: GRPO training with vLLM customer-sim (OpenAI API), Hub persistence, persona gate.
# Run from repository root, OR the script will clone to /workspace/role-drift-env.
# HF auth: prefer env HF_TOKEN / HUGGINGFACE_HUB_TOKEN, else token from `huggingface-cli login` (see training/hf_auth.py).
#
# shellcheck disable=SC1091
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# HF Jobs H200 / NVSwitch: cudaGetDeviceCount can return Error 802 until fabric is ready (see huggingface_hub#4134).
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
# Try 1 first; on H200 some nodes fail NVML path — warmup loop can retry with 0.
export PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Prefer legacy vLLM engine: avoids torch.accelerator worker init issues on some PyTorch + H200 stacks.
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

echo "===== V9 Training Job started at $(date) ====="

REPO_URL="${REPO_URL:-https://github.com/GeniusPlums/OpenEnv-Finale.git}"
REPO_DIR="${REPO_DIR:-/workspace/role-drift-env}"

if [[ ! -f "$REPO_DIR/training/train_grpo.py" ]]; then
  echo "Cloning $REPO_URL -> $REPO_DIR"
  mkdir -p "$(dirname "$REPO_DIR")"
  rm -rf "$REPO_DIR"
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR" || exit 1
export PYTHONPATH="$(pwd)"

# HF `python:3.11` + `pip install -e .` often resolves to CPU-only torch from PyPI; then vLLM keeps it
# and every CUDA probe fails forever. Install CUDA torch first, then the rest.
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e . -q
pip install -q vllm sentence-transformers langdetect peft accelerate bitsandbytes huggingface_hub openai

export HF_USER="${HF_USER:-GeniusPlums}"
export TRAINED_REPO="${TRAINED_REPO:-GeniusPlums/role-drift-qwen-1-5b-grpo}"
# With set -u, use :- on every expansion (bare $HF_TOKEN in :-default still trips unbound)
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"

python scripts/hf_auth_preflight.py || true
# whoami often missing in minimal images even when HF_TOKEN is set; uploads still use the env token.
if huggingface-cli whoami 2>/dev/null; then
  : # ok
elif [[ -n "${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}" ]]; then
  echo "(huggingface-cli whoami not available; HF_TOKEN/HUGGINGFACE_HUB_TOKEN is set — Hub CLI uploads are fine)"
else
  echo "(No HF token in env; run hf auth login locally or set Job secret)"
fi

# --- vLLM customer server (7B) — same card as 1.5B policy: persona must use this API, not a second in-process 7B ---
VLLM_PORT="${VLLM_PORT:-8000}"
export ROLE_DRIFT_PERSONA_OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

echo "===== nvidia-smi ====="
nvidia-smi

# H200 / NVSwitch: touching CUDA before fabric is ready yields Error 802. Do not run torch here.
echo "===== Waiting for GPU fabric (initial 45s after nvidia-smi) ====="
sleep 45

echo "===== CUDA readiness warmup (no set_device — avoids torch 802 on some H200 stacks) ====="
CUDA_OK=0
_WARM_TRIES="${CUDA_WARMUP_TRIES:-48}"
_WARM_SLEEP="${CUDA_WARMUP_SLEEP:-15}"
for i in $(seq 1 "$_WARM_TRIES"); do
  # Alternate NVML check after half the tries (some HF H200 nodes need NVML off).
  if [[ "$i" -eq $((_WARM_TRIES / 2)) ]]; then
    export PYTORCH_NVML_BASED_CUDA_CHECK=0
    echo "[warmup] switching PYTORCH_NVML_BASED_CUDA_CHECK=0 for remaining probes"
  fi
  # Avoid torch.cuda.set_device(0): triggers _cuda_setDevice → 802 while fabric still down.
  _probe_out="$(
    python -c "
import torch
if torch.cuda.device_count() < 1:
    raise RuntimeError('no cuda devices')
x = torch.zeros(1, device='cuda', dtype=torch.float32)
torch.cuda.synchronize()
print('cuda_ok', torch.cuda.get_device_name(0), 'torch', torch.__version__)
del x
" 2>&1
  )" && _probe_rc=0 || _probe_rc=$?
  if [[ "$_probe_rc" -eq 0 ]]; then
    echo "$_probe_out"
    echo "CUDA probe OK after $i attempt(s) (~$((i * ${_WARM_SLEEP}))s budget)"
    CUDA_OK=1
    break
  fi
  echo "CUDA probe $i/$_WARM_TRIES failed:"
  echo "$_probe_out" | head -35
  sleep "$_WARM_SLEEP"
done
if [[ "$CUDA_OK" -ne 1 ]]; then
  echo "FATAL: CUDA not usable after warmup (H200 Error 802 / fabric — re-queue job or use HF_JOB_FLAVOR=a100-large)."
  exit 1
fi

echo "===== CUDA torch sanity (after fabric ready) ====="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'devices', torch.cuda.device_count())"

echo "===== Starting customer-sim vLLM server ====="
# Leave ~35–40GB for policy 1.5B + ref + optimizer states + backward (80GB A100 is tight with 7B vLLM)
# --enforce-eager: avoids CUDA graph capture failures during init on some drivers (vLLM troubleshooting).
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

echo "===== Waiting for vLLM (up to 5 min) ====="
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

echo "===== Verifying customer-sim (live, not fallback) ====="
# shellcheck disable=SC2016
if ! python -c '
import os, sys
assert os.environ.get("ROLE_DRIFT_PERSONA_OPENAI_BASE_URL")
from role_drift_env.server.environment import RoleDriftEnvironment
from role_drift_env.server.personas import load_llm_persona
env = RoleDriftEnvironment()
obs, state = env.reset("term_mu_04")
state.history = [{"role": "agent", "text": "Thanks for your time. Goodbye."}]
p = load_llm_persona("thank_you_bomber")
out = p.next_utterance(state, 42)
print("PERSONA_OUTPUT:", (out or "")[:300])
FALL = "Thanks, I think I have what I need. Goodbye."
is_fb = (FALL in (out or "")) or (len((out or "").strip()) < 2)
print("IS_FALLBACK:", is_fb)
sys.exit(1 if is_fb else 0)
'; then
  echo "FATAL: Persona check failed (IS_FALLBACK True or no output). See above."
  exit 1
fi
echo "===== Persona check PASSED ====="

echo "===== Create private model repo (if needed) ====="
python -m training.hub_upload create-repo "$TRAINED_REPO" || echo "(create-repo note above)"

# V9: same hyperparams; --max-turns 6 caps seq len in backward (avoids OOM with vLLM on one GPU)
echo "===== Starting GRPO training ====="
set +e
python training/train_grpo.py \
  --episodes 100 \
  --group-size 4 \
  --lr 5e-6 \
  --kl-coef 0.125 \
  --lang-term-oversample 1 \
  --curriculum adversarial \
  --policy-model Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint-every 25 \
  --max-turns 6 \
  --output-dir data/training_logs/run_final \
  --checkpoint-dir checkpoints/grpo_final \
  --hub-repo "$TRAINED_REPO" \
  2>&1 | tee training.log
TRAIN_EXIT=$?
set -e
if [[ "$TRAIN_EXIT" -ne 0 ]]; then
  echo "Training exited with code $TRAIN_EXIT; uploading whatever is in best/ and logs (partial run)."
else
  echo "Training finished with exit 0."
fi

echo "===== Push best + logs to Hub (runs even if training crashed) ====="
if [[ -d "checkpoints/grpo_final/best" ]] && [[ -n "$(ls -A checkpoints/grpo_final/best 2>/dev/null || true)" ]]; then
  if [[ "$TRAIN_EXIT" -eq 0 ]]; then
    MSG_BEST="GRPO Qwen 1.5B final best — 100 episodes V9"
  else
    MSG_BEST="Partial/final best checkpoint (training exited $TRAIN_EXIT) V9"
  fi
  python -m training.hub_upload upload-folder "$TRAINED_REPO" checkpoints/grpo_final/best --message "$MSG_BEST" \
    || { echo "WARNING: best upload failed (check HF token / permissions; non-fatal)."; }
else
  echo "WARNING: checkpoints/grpo_final/best is missing or empty. Nothing to upload for weights."
fi

if [[ -f data/training_logs/run_final/episode_log.jsonl ]]; then
  cp -f data/training_logs/run_final/episode_log.jsonl /tmp/episode_log.jsonl
  if [[ "$TRAIN_EXIT" -eq 0 ]]; then
    MSG_LOG="Episode log 100-ep V9"
  else
    MSG_LOG="Episode log partial run V9 (train exit $TRAIN_EXIT)"
  fi
  python -m training.hub_upload upload-file "$TRAINED_REPO" /tmp/episode_log.jsonl --path-in-repo episode_log.jsonl --message "$MSG_LOG" \
    || { echo "WARNING: episode log upload failed (auth?)"; }
else
  echo "WARNING: data/training_logs/run_final/episode_log.jsonl not found; skipping episode log upload."
fi

if [[ -f training.log ]]; then
  cp -f training.log /tmp/training_run.log
  python -m training.hub_upload upload-file "$TRAINED_REPO" /tmp/training_run.log --path-in-repo training.log --message "Full training stdout V9" \
    || { echo "WARNING: training.log upload failed (auth?)"; }
fi

echo "===== Verify read access to model repo (optional) ====="
rm -rf /tmp/hf_verify
mkdir -p /tmp/hf_verify
if python -c "
from huggingface_hub import hf_hub_download
import os
r = os.environ.get('TRAINED_REPO', '')
if not r:
    raise SystemExit(1)
hf_hub_download(repo_id=r, filename='config.json', local_dir='/tmp/hf_verify', repo_type='model')
print('ok')
" 2>/dev/null; then
  ls -la /tmp/hf_verify
else
  echo "WARNING: Hub verify download skipped (token or repo read access)."
fi

echo "===== V9 complete at $(date) ====="
echo "Model: https://huggingface.co/${TRAINED_REPO}"
# Propagate training failure to HF Job status after uploads attempted
exit "$TRAIN_EXIT"
