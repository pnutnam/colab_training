#!/usr/bin/env -S colab run --gpu T4
"""Colab VM smoke test: verify GPU, storage, and Hugging Face connectivity.

Runs on a fresh T4 via `make smoke` (or directly: `./scripts/smoke_test.py`).
Exits non-zero on any failure so `colab run` propagates it.

NOTE: when launched through scripts/launch.sh, an env-injection header is
prepended to this file — the shebang below is inert there (just a comment),
and only matters for direct shebang execution.
"""
import shutil
import subprocess
import sys
import time
import urllib.request

FAILED: list[str] = []


def check(name: str, fn):
    try:
        result = fn()
        print(f"  [ok]   {name}: {result}")
    except Exception as exc:  # noqa: BLE001 - smoke test reports everything
        FAILED.append(name)
        print(f"  [FAIL] {name}: {exc}", file=sys.stderr)


print("== Colab VM smoke test ==")

check("python", lambda: sys.version.split()[0])

check("disk (/content)", lambda: f"{shutil.disk_usage('/content').free / 1e9:.1f} GB free")

check("huggingface.co reachable", lambda: (
    urllib.request.urlopen("https://huggingface.co", timeout=10).status, "HTTP ok"
))


def cuda_check():
    import torch

    assert torch.cuda.is_available(), "CUDA not available — did the VM get a GPU?"
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1e9
    # Timed matmul: rough FLOPS sanity signal.
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tflops = 20 * 2 * 4096**3 / dt / 1e12
    return f"{name}, {vram:.1f} GB VRAM, ~{tflops:.1f} TFLOPS bf16"


check("GPU + CUDA", cuda_check)

def hf_hub_check():
    try:
        import huggingface_hub

        return huggingface_hub.__version__
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True
        )
        import huggingface_hub

        return f"installed {huggingface_hub.__version__}"


check("hf_hub", hf_hub_check)

if FAILED:
    print(f"\nSMOKE TEST FAILED: {FAILED}", file=sys.stderr)
    sys.exit(1)

print("\nAll smoke checks passed — VM is ready for training.")
