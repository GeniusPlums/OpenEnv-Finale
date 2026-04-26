#!/usr/bin/env python3
"""Active wait until PyTorch can initialize CUDA (HF Jobs / H200 fabric race).

Uses a **fresh Python subprocess** each attempt so a failed cuda init does not poison
the interpreter. Exit 0 when ready, 1 after max retries (~5 min default).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

# One short process: init + tiny allocation (matches training / vLLM needs).
# Do NOT call device_count() first — on HF H200 / NVSwitch, cudaGetDeviceCount can throw
# Error 802 until the fabric is ready; binding the device + current_device() matches a
# more reliable init order than count-first.
_PROBE = r"""
import torch

def _probe():
    # Explicit init (no-op if already done); can succeed slightly before count-first path stabilizes.
    try:
        torch.cuda.init()
    except Exception as e:
        raise RuntimeError(f"torch.cuda.init failed: {e}") from e
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False after init")
    torch.cuda.set_device(0)
    torch.cuda.current_device()
    n = torch.cuda.device_count()
    if n < 1:
        raise RuntimeError("no CUDA devices")
    name = torch.cuda.get_device_name(0)
    x = torch.zeros(2, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    del x
    print("cuda_ok", name, "torch", torch.__version__, "devices", n)

_probe()
"""


def main() -> int:
    max_r = int(os.environ.get("CUDA_WAIT_MAX_RETRIES", "60"))
    sleep_s = float(os.environ.get("CUDA_WAIT_SLEEP_SEC", "5"))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")

    print(
        f"[wait_for_cuda] max_retries={max_r} sleep_sec={sleep_s} "
        f"(~{max_r * sleep_s:.0f}s cap); subprocess per attempt",
        flush=True,
    )

    initial = float(os.environ.get("CUDA_WAIT_INITIAL_SLEEP_SEC", "0"))
    if initial > 0:
        print(f"[wait_for_cuda] initial sleep {initial}s (CUDA_WAIT_INITIAL_SLEEP_SEC)", flush=True)
        time.sleep(initial)

    for i in range(max_r):
        if i == max_r // 2:
            os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
            print("[wait_for_cuda] switching PYTORCH_NVML_BASED_CUDA_CHECK=1", flush=True)

        proc = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=120,
        )
        if proc.returncode == 0:
            out = (proc.stdout or "").strip()
            if out:
                print(out, flush=True)
            print(f"[wait_for_cuda] CUDA ready after {i + 1} attempt(s)", flush=True)
            return 0

        err = (proc.stderr or proc.stdout or "").strip()
        tail = err.splitlines()[-1] if err else f"exit {proc.returncode}"
        print(f"[wait_for_cuda] not ready ({i + 1}/{max_r}): {tail}", flush=True)
        time.sleep(sleep_s)

    print("[wait_for_cuda] FATAL: CUDA never became ready", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
