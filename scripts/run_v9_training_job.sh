#!/usr/bin/env bash
# V9: GRPO training with vLLM customer-sim (OpenAI API), Hub persistence, persona gate.
# Run from repository root, OR the script will clone to /workspace/role-drift-env.
# HF auth: prefer env HF_TOKEN / HUGGINGFACE_HUB_TOKEN, else token from `huggingface-cli login` (see training/hf_auth.py).
#
# shellcheck disable=SC1091
set -euo pipefail

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

pip install -e . -q
pip install -q vllm sentence-transformers langdetect peft accelerate bitsandbytes huggingface_hub openai

export HF_USER="${HF_USER:-GeniusPlums}"
export TRAINED_REPO="${TRAINED_REPO:-GeniusPlums/role-drift-qwen-1-5b-grpo}"
# With set -u, use :- on every expansion (bare $HF_TOKEN in :-default still trips unbound)
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"

python scripts/hf_auth_preflight.py || true
huggingface-cli whoami 2>/dev/null || echo "(Hub CLI: not logged in — Hub uploads may fail until token is available)"

# --- vLLM customer server (7B) — same card as 1.5B policy: persona must use this API, not a second in-process 7B ---
VLLM_PORT="${VLLM_PORT:-8000}"
export ROLE_DRIFT_PERSONA_OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

echo "===== nvidia-smi ====="
nvidia-smi

echo "===== Starting customer-sim vLLM server ====="
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port "$VLLM_PORT" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.40 \
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
huggingface-cli repo create "$TRAINED_REPO" --type model --private 2>/dev/null || echo "(repo may already exist; continuing)"

# OPENCODE V9: lr 1e-5, kl 0.05, 100 ep, no max-turns. Add --max-turns 6 here only if you hit policy OOM.
echo "===== Starting GRPO training ====="
set +e
python training/train_grpo.py \
  --episodes 100 \
  --group-size 4 \
  --lr 1e-5 \
  --kl-coef 0.05 \
  --curriculum adversarial \
  --policy-model Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint-every 25 \
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
  huggingface-cli upload "$TRAINED_REPO" checkpoints/grpo_final/best \
    --repo-type model \
    --commit-message "$MSG_BEST" || { echo "WARNING: best upload failed (check HF token / permissions; non-fatal)."; }
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
  huggingface-cli upload "$TRAINED_REPO" /tmp/episode_log.jsonl --commit-message "$MSG_LOG" || { echo "WARNING: episode log upload failed (auth?)"; }
else
  echo "WARNING: data/training_logs/run_final/episode_log.jsonl not found; skipping episode log upload."
fi

if [[ -f training.log ]]; then
  cp -f training.log /tmp/training_run.log
  huggingface-cli upload "$TRAINED_REPO" /tmp/training_run.log --commit-message "Full training stdout V9" || { echo "WARNING: training.log upload failed (auth?)"; }
fi

echo "===== Verify download (optional; needs read access to repo) ====="
rm -rf /tmp/hf_verify
if huggingface-cli download "$TRAINED_REPO" --include="config.json" --quiet --local-dir /tmp/hf_verify 2>/dev/null; then
  ls -la /tmp/hf_verify
else
  echo "WARNING: Hub verify download skipped or failed (missing token or repo not visible)."
fi

echo "===== V9 complete at $(date) ====="
echo "Model: https://huggingface.co/${TRAINED_REPO}"
# Propagate training failure to HF Job status after uploads attempted
exit "$TRAIN_EXIT"
