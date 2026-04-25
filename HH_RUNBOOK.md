# HH_RUNBOOK.md — Pre-flight on real GPU runtime

Goal: clear HH steps 4 and 6 on a real GPU runtime before provisioning paid A100.

## Colab setup

```bash
!git clone <YOUR_REPO_URL> /content/role-drift-env
%cd /content/role-drift-env
!pip install -e .
!pip install vllm sentence-transformers langdetect peft accelerate bitsandbytes
!python -c "import vllm; print(vllm.__version__)"
```

## Step 2 — vLLM customer-sim check

Start vLLM:

```bash
!python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.6 &
!sleep 90
!curl http://localhost:8000/v1/models
```

Probe persona wiring (current class name is `LLMPersona`):

```bash
!python -c "
from role_drift_env.server.personas.llm_backed import LLMPersona
from role_drift_env.server.customer_sim import CustomerSimulator
from role_drift_env.server.environment import RoleDriftEnvironment
env = RoleDriftEnvironment()
obs, state = env.reset('term_kk_01')
p = LLMPersona(persona_id='hh_probe', system_prompt='You are a polite customer.', model='Qwen/Qwen2.5-7B-Instruct')
out = p.next_utterance(state, rng_seed=42)
print('PERSONA OUTPUT:', out)
print('IS_SCRIPTED_FALLBACK:', 'Thanks, I think I have what I need' in out)
"
```

Acceptance: vLLM server responds and `IS_SCRIPTED_FALLBACK: False`.

## Step 3 — real-model timing smoke

```bash
!python training/train_grpo.py \
  --episodes 2 \
  --group-size 2 \
  --lr 1e-5 \
  --kl-coef 0.05 \
  --curriculum adversarial \
  --policy-model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir /tmp/smoke_real \
  --checkpoint-dir /tmp/smoke_real_ckpt \
  --checkpoint-every 1 \
  --max-turns 8 \
  --time-each-episode
```

Acceptance:
- 2 episodes complete
- KL non-zero and no NaN
- per-component fields present in `/tmp/smoke_real/episode_log.jsonl`
- per-episode timing printed (`--time-each-episode`)

## Step 4 — HH gate decision matrix

| HH item | Status |
|---|---|
| 1. Smoke train | PASS (local) |
| 2. Smoke eval baseline | PASS (local) |
| 3. Smoke eval all scenario sets | PASS (local) |
| 4. vLLM customer-sim | Fill from Step 2 |
| 5. Checkpoint save/load round-trip | PASS (local) |
| 6. Real-model timing | Fill from Step 3 |

GO only if all PASS and projected training cost <= $15.

