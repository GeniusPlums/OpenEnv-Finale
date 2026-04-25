"""Entry for shell jobs: `python scripts/hf_auth_preflight.py` (repo root on PYTHONPATH)."""
from training.hf_auth import run_preflight

if __name__ == "__main__":
    run_preflight()
