"""
Launch V9 training on Hugging Face Jobs (A100) — run_v9_training_job.sh (vLLM + Hub persistence).

**HF_TOKEN in the job container**
- **This launcher (recommended):** set `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) in your *local* environment
  before running this script. It is passed via the Jobs API `secrets` parameter (not written to the repo).
- **Web UI jobs:** add a secret/mapping so the container has `HF_TOKEN` set to your write token.

Do not commit tokens. If a token was ever pasted into chat, revoke it and create a new one.

Timeout default: 2h. Override with HF_JOB_TIMEOUT=120m
"""
from huggingface_hub import HfApi
import os


def main() -> None:
    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf:
        print(
            "WARNING: HF_TOKEN not set locally; job container will use hf_auth_preflight only. "
            "Fresh HF Jobs usually need a secret: set HF_TOKEN here or in the Job UI."
        )

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
exec bash scripts/run_v9_training_job.sh
""".strip()

    run_kw = {
        "image": "python:3.11",
        "command": ["bash", "-lc", bash_cmd],
        "flavor": flavor,
        "timeout": timeout,
        "namespace": namespace,
        "token": True,
    }
    if hf:
        run_kw["secrets"] = {"HF_TOKEN": hf, "HUGGINGFACE_HUB_TOKEN": hf}

    info = api.run_job(**run_kw)
    print(info.id)
    print(info.url)


if __name__ == "__main__":
    main()
