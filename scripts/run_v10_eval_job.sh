#!/usr/bin/env bash
# V10 eval: wrapper clones repo, then runs run_v10_eval_worker.sh with full-process restarts
# on CUDA/vLLM failure (new bash = clean interpreter; same container may still have bad GPU).
#
# shellcheck disable=SC1091
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

echo "V10_ENTRY ROLE_DRIFT_PERSONA_OPENAI_BASE_URL=${ROLE_DRIFT_PERSONA_OPENAI_BASE_URL-}"

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

export V10_REPO_ROOT="$(pwd)"
export PYTHONPATH="$(pwd)"
echo "===== V10 Eval wrapper at $(date) in $V10_REPO_ROOT ====="

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "FATAL: HF_TOKEN unset. Aborting before spend."
  exit 1
fi

EVAL_FULL_RESTARTS="${EVAL_FULL_RESTARTS:-3}"
FULL_RESTART_COOLDOWN="${FULL_RESTART_COOLDOWN:-20}"

cleanup_gpu_children() {
  echo "===== [wrapper] cleaning up vLLM / stray python GPU processes ====="
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -f "python -m vllm" 2>/dev/null || true
  sleep 2
}

for restart in $(seq 1 "$EVAL_FULL_RESTARTS"); do
  export V10_EVAL_RESTART="$restart"
  echo "===== Full eval process attempt ${restart}/${EVAL_FULL_RESTARTS} (new bash worker) ====="
  if bash "$V10_REPO_ROOT/scripts/run_v10_eval_worker.sh"; then
    echo "===== V10 eval succeeded on attempt ${restart} ====="
    exit 0
  fi
  _ec=$?
  echo "===== Worker exited ${_ec} (CUDA fatal=2, sanity=3). ====="
  cleanup_gpu_children
  if [[ "$restart" -lt "$EVAL_FULL_RESTARTS" ]]; then
    echo "===== Full restart in ${FULL_RESTART_COOLDOWN}s (EVAL_FULL_RESTARTS) ====="
    sleep "$FULL_RESTART_COOLDOWN"
  fi
done

echo "FATAL: V10 eval failed after ${EVAL_FULL_RESTARTS} full worker attempts. Re-queue HF Job or change hardware."
exit 1
