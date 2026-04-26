# HH_RUNBOOK_HF.md — Paid Mini-Smoke on HF, Then Go

Goal: clear HH steps 4 and 6 in ~15-25 minutes on paid HF A100, then start Section II in the same runtime.

## 0) Pre-flight (local, $0)

- Repo committed + pushed so HF can clone latest state.
- Local HH steps 1, 2, 3, 5 are already PASS.
- HF auth ready: `HF_TOKEN` set or token saved with `huggingface-cli login`.
- Budget alert set before $30 cap.

## 1) Provisioning

Use **A100 40GB** (~$3-4/hr). Do not use H100/T4/L4 for this gate.

## 2) Mini-smoke (A100, ~$1-2)

```bash
# Clone + install
git clone https://github.com/GeniusPlums/OpenEnv-Finale.git /workspace/role-drift-env
cd /workspace/role-drift-env
pip install -e .
pip install vllm sentence-transformers langdetect peft accelerate bitsandbytes

# Verify GPU
nvidia-smi

# Start vLLM customer-sim backend
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.45 > /tmp/vllm.log 2>&1 &
sleep 120
curl -s http://localhost:8000/v1/models

# Persona check (current class is LLMPersona in this repo)
python -c "
from role_drift_env.server.personas.llm_backed import LLMPersona
from role_drift_env.server.environment import RoleDriftEnvironment
env = RoleDriftEnvironment()
obs, state = env.reset('term_kk_01')
p = LLMPersona(persona_id='hh_probe', system_prompt='You are a polite customer.', model='Qwen/Qwen2.5-7B-Instruct')
out = p.next_utterance(state, rng_seed=42)
print('PERSONA OUTPUT:', out[:200])
print('IS_FALLBACK:', 'I think I have what I need' in out)
"

# 2-episode real timing smoke (same recipe knobs)
time python training/train_grpo.py \
  --episodes 2 \
  --group-size 4 \
  --lr 1e-5 \
  --kl-coef 0.05 \
  --curriculum adversarial \
  --policy-model Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint-every 1 \
  --output-dir /tmp/smoke_real \
  --checkpoint-dir /tmp/smoke_ckpt \
  --time-each-episode
```

## 3) Decision gate

Record per-episode time `T` from log (`Episode wall time`).

- Projected 100-episode time: `100 * T`
- Projected cost: `(100 * T / 3600) * $4`

Decision:
- `<= $12`: GO at 100 episodes
- `$12-$18`: GO at 70-80 episodes
- `> $18`: NO-GO, diagnose before spending more

Also require:
- vLLM `/v1/models` returns model list
- persona output is non-fallback (`IS_FALLBACK: False`)
- KL non-zero, no NaN, per-component fields present in `episode_log.jsonl`

## 4) If GO: continue in same runtime (recommended)

```bash
rm -rf /tmp/smoke_real /tmp/smoke_ckpt

python training/train_grpo.py \
  --episodes 100 \
  --group-size 4 \
  --lr 1e-5 \
  --kl-coef 0.05 \
  --curriculum adversarial \
  --policy-model Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint-every 25 \
  --output-dir data/training_logs/run_final \
  --checkpoint-dir checkpoints/grpo_final 2>&1 | tee training.log
```

Live babysit:
- `tail -f training.log`
- check `episode_log.jsonl` at episode 25/50/75
- stop on KL collapse/explosion or reward collapse

## 5) Quick troubleshooting

- vLLM OOM on startup: reduce `--gpu-memory-utilization` to `0.35`
- vLLM hangs: restart once, then consider temporary hosted customer backend
- Fallback output persists: verify persona path and local vLLM readiness from `/tmp/vllm.log`

