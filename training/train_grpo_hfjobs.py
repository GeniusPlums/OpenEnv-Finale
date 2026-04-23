#!/usr/bin/env python3
"""
HF Jobs entrypoint for GRPO training.

Usage:
    hf jobs run --flavor l40sx1 --secrets HF_TOKEN python:3.12 python training/train_grpo_hfjobs.py --episodes 5 ...

This is a wrapper around train_grpo.py that:
1. Accepts identical CLI args
2. Runs training locally
3. Uploads checkpoints + logs to output repo when done
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN")
OUTPUT_REPO = os.environ.get("OUTPUT_REPO", "GeniusPlums/role-drift-runs")


def upload_results(checkpoint_dir: Path, output_repo: str, token: str):
    """Upload checkpoint directory to HF Hub."""
    if not token:
        print("[HFJOBS] HF_TOKEN not set, skipping upload")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    print(f"[HFJOBS] Uploading checkpoints from {checkpoint_dir} to {output_repo}")

    try:
        api.upload_folder(
            folder_path=str(checkpoint_dir),
            repo_id=output_repo,
            repo_type="model",
            commit_message=f"Training run checkpoint",
            token=token,
        )
        print(f"[HFJOBS] Upload complete: https://huggingface.co/{output_repo}")
    except Exception as e:
        print(f"[HFJOBS] Upload failed: {e}")
        raise


def main():
    """Run the GRPO training and upload results."""
    import argparse

    parser = argparse.ArgumentParser(description="HF Jobs GRPO trainer")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--scenario-file", default="data/scenarios/train.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/grpo")
    parser.add_argument("--output-repo", default=OUTPUT_REPO)
    parser.add_argument("--sft-checkpoint", default="checkpoints/sft")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-transcripts-every", type=int, default=5)
    parser.add_argument("--transcript-dir", default="logs/transcripts")
    parser.add_argument("--skip-upload", action="store_true", help="Skip uploading results")

    args = parser.parse_args()

    print(f"[HFJOB] Starting GRPO training on HF Jobs")
    print(f"[HFJOB] Episodes: {args.episodes}")
    print(f"[HFJOB] Group size: {args.group_size}")
    print(f"[HFJOB] Model: {args.model_name}")
    print(f"[HFJOB] KL coef: {args.kl_coef}")
    print(f"[HFJOB] LR: {args.lr}")
    print(f"[HFJOB] Output repo: {args.output_repo}")

    # Build command to run train_grpo.py
    cmd = [
        sys.executable,
        "training/train_grpo.py",
        "--episodes", str(args.episodes),
        "--group-size", str(args.group_size),
        "--kl-coef", str(args.kl_coef),
        "--lr", str(args.lr),
        "--model-name", args.model_name,
        "--scenario-file", args.scenario_file,
        "--output-dir", args.output_dir,
        "--log-every", str(args.log_every),
        "--save-transcripts-every", str(args.save_transcripts_every),
        "--transcript-dir", args.transcript_dir,
    ]

    # Add SFT checkpoint if exists
    if args.sft_checkpoint and Path(args.sft_checkpoint).exists():
        cmd.extend(["--sft-checkpoint", args.sft_checkpoint])

    print(f"[HFJOB] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path.cwd())

    if result.returncode != 0:
        print(f"[HFJOB] Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"[HFJOB] Training complete. Exit code: {result.returncode}")

    if args.skip_upload:
        print("[HFJOB] Skipping upload (--skip-upload set)")
    else:
        checkpoint_dir = Path(args.output_dir)
        if checkpoint_dir.exists():
            upload_results(checkpoint_dir, args.output_repo, HF_TOKEN)
        else:
            print(f"[HFJOB] Warning: output dir {checkpoint_dir} not found, skipping upload")

    print("[HFJOB] Done.")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()