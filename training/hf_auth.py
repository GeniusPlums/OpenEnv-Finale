"""Hugging Face token resolution and optional login. Never exit on missing token."""
from __future__ import annotations

import os
from typing import Optional


def resolve_hf_token() -> Optional[str]:
    """Env first, then cached token from `huggingface-cli login` / Hf storage."""
    t = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if t:
        return t
    try:
        from huggingface_hub import get_token

        t2 = get_token()
        return t2.strip() if t2 else None
    except Exception:
        pass
    # Older hub: HfFolder (may be absent on newer versions)
    try:
        from huggingface_hub import HfFolder  # type: ignore

        t3 = HfFolder.get_token()
        return t3.strip() if t3 else None
    except Exception:
        return None


def run_preflight() -> Optional[str]:
    """
    If a token is found, call huggingface_hub.login. Always print auth status.
    Returns resolved token or None. Never raises for missing token.
    """
    token = resolve_hf_token()
    if token:
        try:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)
        except Exception as e:
            print(f"WARNING: huggingface_hub.login failed (continuing): {e}")
    else:
        print("WARNING: No Hugging Face token found. Proceeding without authentication.")
    print("HF auth status:", "available" if token else "missing")
    return token


if __name__ == "__main__":
    run_preflight()
