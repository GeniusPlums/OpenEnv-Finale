"""Hub uploads using huggingface_hub (avoids deprecated huggingface-cli in newer hub releases)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi

from training.hf_auth import resolve_hf_token


def _api(token: Optional[str] = None) -> Optional[HfApi]:
    t = token or resolve_hf_token()
    if not t:
        return None
    return HfApi(token=t)


def ensure_model_repo(repo_id: str) -> bool:
    """Create private model repo if missing. Returns True if ok or already exists."""
    api = _api()
    if not api:
        print("[hub_upload] No token; skip create_repo")
        return False
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=True,
            exist_ok=True,
        )
        return True
    except Exception as e:
        print(f"[hub_upload] create_repo: {e}")
        return False


def upload_model_folder(
    repo_id: str,
    folder_path: str | Path,
    commit_message: str,
) -> bool:
    folder_path = Path(folder_path)
    if not folder_path.is_dir() or not any(folder_path.iterdir()):
        print(f"[hub_upload] No folder to upload: {folder_path}")
        return False
    api = _api()
    if not api:
        print("[hub_upload] No token; cannot upload")
        return False
    try:
        api.upload_folder(
            folder_path=str(folder_path),
            path_in_repo=None,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        print(f"[hub_upload] OK: {folder_path} -> {repo_id}")
        return True
    except Exception as e:
        print(f"[hub_upload] upload_folder failed: {e}")
        return False


def upload_file(
    repo_id: str,
    local_path: str | Path,
    path_in_repo: str,
    commit_message: str,
) -> bool:
    local_path = Path(local_path)
    if not local_path.is_file():
        print(f"[hub_upload] Missing file: {local_path}")
        return False
    api = _api()
    if not api:
        print("[hub_upload] No token; cannot upload")
        return False
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        print(f"[hub_upload] OK: {path_in_repo} -> {repo_id}")
        return True
    except Exception as e:
        print(f"[hub_upload] upload_file failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Hub uploads (HfApi)")
    sub = p.add_subparsers(dest="cmd", required=True)
    a0 = sub.add_parser("create-repo", help="Create private model repo if missing")
    a0.add_argument("repo_id")
    a1 = sub.add_parser("upload-folder")
    a1.add_argument("repo_id")
    a1.add_argument("folder")
    a1.add_argument("--message", default="upload")
    a2 = sub.add_parser("upload-file")
    a2.add_argument("repo_id")
    a2.add_argument("path")
    a2.add_argument("--path-in-repo", required=True)
    a2.add_argument("--message", default="upload")
    args = p.parse_args()
    if args.cmd == "create-repo":
        ok = ensure_model_repo(args.repo_id)
        sys.exit(0 if ok else 1)
    if args.cmd == "upload-folder":
        ok = upload_model_folder(args.repo_id, args.folder, args.message)
        sys.exit(0 if ok else 1)
    if args.cmd == "upload-file":
        ok = upload_file(args.repo_id, args.path, args.path_in_repo, args.message)
        sys.exit(0 if ok else 1)
