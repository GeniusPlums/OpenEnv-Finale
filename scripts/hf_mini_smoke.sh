#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/GeniusPlums/OpenEnv-Finale.git}"
WORKDIR="${WORKDIR:-/workspace/role-drift-env}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
POLICY_MODEL="${POLICY_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

echo "[1/7] Clone + install"
rm -rf "$WORKDIR"
git clone "$REPO_URL" "$WORKDIR"
cd "$WORKDIR"
pip install -e .
pip install vllm sentence-transformers langdetect peft accelerate bitsandbytes

echo "[2/7] GPU info"
nvidia-smi

echo "[3/7] Start vLLM"
pkill -f vllm.entrypoints.openai.api_server || true
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$VLLM_MODEL" \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.45 > /tmp/vllm.log 2>&1 &
sleep 120

echo "[4/7] Check vLLM /v1/models"
curl -s http://localhost:8000/v1/models | python -m json.tool

echo "[5/7] Persona fallback check"
python - <<'PY'
from role_drift_env.server.personas.llm_backed import LLMPersona
from role_drift_env.server.environment import RoleDriftEnvironment
env = RoleDriftEnvironment()
obs, state = env.reset("term_kk_01")
p = LLMPersona(persona_id="hh_probe", system_prompt="You are a polite customer.", model="Qwen/Qwen2.5-7B-Instruct")
out = p.next_utterance(state, rng_seed=42)
print("PERSONA OUTPUT:", out[:200])
print("IS_FALLBACK:", "I think I have what I need" in out)
PY

echo "[6/7] 2-episode timing smoke"
time python training/train_grpo.py \
  --episodes 2 \
  --group-size 4 \
  --lr 1e-5 \
  --kl-coef 0.05 \
  --curriculum adversarial \
  --policy-model "$POLICY_MODEL" \
  --checkpoint-every 1 \
  --output-dir /tmp/smoke_real \
  --checkpoint-dir /tmp/smoke_ckpt \
  --time-each-episode

echo "[7/7] Summarize smoke outputs"
python - <<'PY'
import json, pathlib
p = pathlib.Path("/tmp/smoke_real/episode_log.jsonl")
rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
times = [r.get("episode_seconds", 0.0) for r in rows]
mean_t = sum(times)/len(times)
cost = (100*mean_t/3600)*4.0
print(f"episodes={len(rows)} mean_episode_seconds={mean_t:.2f}")
print(f"projected_cost_100ep_usd={cost:.2f}")
print(f"avg_kl={sum(r.get('avg_kl',0.0) for r in rows)/len(rows):.4f}")
print(f"component_means_present={all('component_means' in r for r in rows)}")
PY

echo "Mini-smoke complete."
