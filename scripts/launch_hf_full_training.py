"""Launch Section II: 100-episode GRPO on Hugging Face Jobs (A100), same stack as launch_hf_smoke_job.

Uses --lr 5e-6 and --kl-coef 0.1 (vs 1e-5 / 0.05) to reduce PPO/GRPO-style KL spikes
that stopped an earlier 100-ep run at ep7 (approx_kl~6 on scenario coop_mu_02).
"""

from huggingface_hub import HfApi
import os


def main() -> None:
    api = HfApi(token=True)
    flavor = os.getenv("HF_JOB_FLAVOR", "a100-large")
    # ~62 min wall @ ~37s/episode + setup + headroom; increase if you see SIGTERM.
    timeout = os.getenv("HF_JOB_TIMEOUT", "300m")
    namespace = os.getenv("HF_JOB_NAMESPACE", "GeniusPlums")

    bash_cmd = r"""
set -euo pipefail
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

apt-get update -y >/dev/null
apt-get install -y git curl >/dev/null
git clone https://github.com/GeniusPlums/OpenEnv-Finale.git /workspace/role-drift-env
cd /workspace/role-drift-env
export PYTHONPATH=/workspace/role-drift-env

pip install -e .
pip install vllm sentence-transformers langdetect peft accelerate bitsandbytes
nvidia-smi

export ROLE_DRIFT_VLLM_GPU_UTIL=0.35
export ROLE_DRIFT_VLLM_MAX_MODEL_LEN=2048

cat > /tmp/persona_check.py <<'PY'
from role_drift_env.server.personas.llm_backed import LLMPersona
from role_drift_env.server.environment import RoleDriftEnvironment

env = RoleDriftEnvironment()
obs, state = env.reset("term_kk_01")
p = LLMPersona(
    persona_id="hh_probe",
    system_prompt="You are a polite customer.",
    model="Qwen/Qwen2.5-7B-Instruct",
)
out = p.next_utterance(state, rng_seed=42)
print("PERSONA_OUTPUT:", out[:200])
print("IS_FALLBACK:", "I think I have what I need" in out)
PY
python /tmp/persona_check.py

rm -rf /tmp/smoke_real /tmp/smoke_ckpt

python training/train_grpo.py \
  --episodes 100 \
  --group-size 4 \
  --lr 5e-6 \
  --kl-coef 0.1 \
  --curriculum adversarial \
  --policy-model Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint-every 25 \
  --max-turns 6 \
  --output-dir data/training_logs/run_final \
  --checkpoint-dir checkpoints/grpo_final \
  2>&1 | tee training.log
""".strip()

    info = api.run_job(
        image="python:3.11",
        command=["bash", "-lc", bash_cmd],
        flavor=flavor,
        timeout=timeout,
        namespace=namespace,
        token=True,
    )
    print(info.id)
    print(info.url)


if __name__ == "__main__":
    main()
