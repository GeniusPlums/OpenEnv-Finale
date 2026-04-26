"""
Launch V10 eval (run_v10_eval_job.sh) on Hugging Face Jobs — Bundle A.

Clone from GitHub, run baseline/trained + transfer + reward probes, upload to Hub.

**Auth:** After `hf auth login`, run this script; the token is injected as a job secret
(same pattern as launch_v9_job.py).

Optional env:
  HF_JOB_FLAVOR   default: h200 (141GB; override with a100-large etc.)
  HF_JOB_TIMEOUT  default: 3h
  HF_JOB_NAMESPACE default: GeniusPlums
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.hf_auth import resolve_hf_token  # noqa: E402


def main() -> None:
    hf = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip() or resolve_hf_token()
    if hf:
        print("HF auth: token from env or `hf auth login` cache -> job secret.")
    else:
        print("WARNING: No token. Run `hf auth login` and retry.")
        sys.exit(1)

    api = HfApi(token=True)
    flavor = os.getenv("HF_JOB_FLAVOR", "h200")
    timeout = os.getenv("HF_JOB_TIMEOUT", "3h")
    namespace = os.getenv("HF_JOB_NAMESPACE", "GeniusPlums")

    bash_cmd = r"""
set -euo pipefail
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
apt-get update -y >/dev/null
apt-get install -y git curl >/dev/null
git clone --depth 1 https://github.com/GeniusPlums/OpenEnv-Finale.git /workspace/role-drift-env
cd /workspace/role-drift-env
export PYTHONPATH=/workspace/role-drift-env
exec bash scripts/run_v10_eval_job.sh
""".strip()

    run_kw: dict = {
        "image": "python:3.11",
        "command": ["bash", "-lc", bash_cmd],
        "flavor": flavor,
        "timeout": timeout,
        "namespace": namespace,
        "token": True,
        "secrets": {"HF_TOKEN": hf, "HUGGINGFACE_HUB_TOKEN": hf},
    }

    print(f"Submitting V10 eval job: flavor={flavor} timeout={timeout} namespace={namespace}")
    info = api.run_job(**run_kw)
    print(info.id)
    print(info.url)


if __name__ == "__main__":
    main()
