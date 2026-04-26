"""Launch V9 GRPO training on Hugging Face Jobs (vLLM customer server, Hub upload).

Runs `scripts/run_v9_training_job.sh` on a fresh clone — same stack as documented in-repo.
Requires HF Jobs (paid) and a write-capable token: `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`,
or a cached token from `hf auth login` / `huggingface_hub.login`.
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import HfApi, get_token


def main() -> None:
    token_val = (
        (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
        or (get_token() or "").strip()
    )
    if not token_val:
        print(
            "FATAL: No Hugging Face token found. Run `hf auth login` or set HF_TOKEN.",
            file=sys.stderr,
        )
        sys.exit(1)

    api = HfApi(token=True)
    # Default h200 (141GB). Warmup + vLLM flags in run_v9_training_job.sh mitigate CUDA 802 on some nodes.
    flavor = os.getenv("HF_JOB_FLAVOR", "h200")
    # Long default: H200 fabric warmup can exceed ~35 min before vLLM starts.
    timeout = os.getenv("HF_JOB_TIMEOUT", "480m")
    namespace = os.getenv("HF_JOB_NAMESPACE", "GeniusPlums")
    repo_url = os.getenv("REPO_URL", "https://github.com/GeniusPlums/OpenEnv-Finale.git")

    # Secrets inject HF_TOKEN into the job env; do not embed the token in `command` text.
    bash_cmd = rf"""
set -euo pipefail
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HUGGINGFACE_HUB_TOKEN="${{HF_TOKEN:-}}"

apt-get update -y >/dev/null
apt-get install -y git curl >/dev/null

rm -rf /workspace/role-drift-env
git clone --depth 1 "{repo_url}" /workspace/role-drift-env
cd /workspace/role-drift-env
export PYTHONPATH="$(pwd)"

bash scripts/run_v9_training_job.sh
""".strip()

    info = api.run_job(
        image="python:3.11",
        command=["bash", "-lc", bash_cmd],
        flavor=flavor,
        timeout=timeout,
        namespace=namespace,
        secrets={"HF_TOKEN": token_val},
        token=True,
    )
    print(info.id)
    print(info.url)


if __name__ == "__main__":
    main()
