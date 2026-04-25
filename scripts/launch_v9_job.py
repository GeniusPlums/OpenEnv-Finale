"""
Launch V9 training on Hugging Face Jobs (A100) — run_v9_training_job.sh (vLLM + Hub persistence).

Requires: huggingface_hub, `huggingface-cli login` or HF env auth.
To pass HF_TOKEN into the job, set the secret in the HF Job UI, or (API) set env in your account.

Timeout default: 2h. Override with HF_JOB_TIMEOUT=120m
"""
from huggingface_hub import HfApi
import os


def main() -> None:
    api = HfApi(token=True)
    flavor = os.getenv("HF_JOB_FLAVOR", "a100-large")
    timeout = os.getenv("HF_JOB_TIMEOUT", "120m")
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
# HF Job secret must be named HF_TOKEN in the UI; CLI accepts token from `huggingface-cli login` on job owner.
exec bash scripts/run_v9_training_job.sh
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
