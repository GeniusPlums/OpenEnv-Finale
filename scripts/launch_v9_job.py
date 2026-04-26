"""
Launch V9 training on Hugging Face Jobs (A100) — run_v9_training_job.sh (vLLM + Hub persistence).

**Auth (no Job UI env vars required):** run this from your machine after `hf auth login`.
The launcher reads the same token the CLI stores on disk (`resolve_hf_token`) and injects it
into the remote job via the API `secrets` field — the container never needs you to type env
vars in the Hugging Face web UI.

Optional: set `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` to override the cached token.

Timeout default: 2h. Override with HF_JOB_TIMEOUT=120m
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

# Repo root on path (so `training.hf_auth` imports when cwd is not the project)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.hf_auth import resolve_hf_token  # noqa: E402


def main() -> None:
    hf = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip() or resolve_hf_token()
    if hf:
        print(
            "HF auth: using token from env or local `hf auth login` cache; injecting into job as secret."
        )
    else:
        print(
            "WARNING: No token in env or local cache. Run `hf auth login`, then run this script again "
            "(Job UI env vars are not required when launching this way)."
        )

    api = HfApi(token=True)
    flavor = os.getenv("HF_JOB_FLAVOR", "h200")
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
