from huggingface_hub import HfApi
import os


def main() -> None:
    api = HfApi(token=True)
    flavor = os.getenv("HF_JOB_FLAVOR", "l40sx1")
    timeout = os.getenv("HF_JOB_TIMEOUT", "45m")
    namespace = os.getenv("HF_JOB_NAMESPACE", "GeniusPlums")

    bash_cmd = r"""
set -euo pipefail
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

apt-get update -y >/dev/null
apt-get install -y git >/dev/null
git clone https://github.com/GeniusPlums/OpenEnv-Finale.git /workspace/role-drift-env
cd /workspace/role-drift-env
export PYTHONPATH=/workspace/role-drift-env

pip install -e .
pip install sentence-transformers langdetect peft accelerate bitsandbytes
nvidia-smi

time python training/train_grpo.py \
  --episodes 2 \
  --group-size 4 \
  --lr 1e-5 \
  --kl-coef 0.05 \
  --curriculum adversarial \
  --policy-model Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint-every 1 \
  --max-turns 6 \
  --output-dir /tmp/smoke_real \
  --checkpoint-dir /tmp/smoke_ckpt \
  --time-each-episode

python - <<'PY'
import json
from pathlib import Path

p = Path("/tmp/smoke_real/episode_log.jsonl")
rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
t = sum(r.get("episode_seconds", 0.0) for r in rows) / len(rows)
cost = (100 * t / 3600) * 4.0
print(f"episodes={len(rows)} mean_episode_seconds={t:.2f}")
print(f"projected_cost_100ep_usd={cost:.2f}")
print(f"avg_kl={sum(r.get('avg_kl', 0.0) for r in rows) / len(rows):.4f}")
print(f"component_means_present={all('component_means' in r for r in rows)}")
PY
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
