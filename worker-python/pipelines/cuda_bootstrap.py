"""CUDA sanity checks and model-ready logging (no pipeline-specific imports)."""

from __future__ import annotations

import os

from rich.console import Console

console = Console()


def _ensure_pytorch_cuda_kernels_work() -> None:
    """Fail fast if PyTorch cannot execute on ``cuda:0`` (common on very new GPUs / wheel mismatch).

    Ultralytics (scene YOLO + license-plate YOLO) uses PyTorch CUDA. Without this check,
    a typical failure is ``cudaErrorNoKernelImageForDevice`` mid-run, then Paddle teardown aborts.
    """
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    try:
        x = torch.randn(32, 32, device="cuda", dtype=torch.float32)
        _ = x @ x
        torch.cuda.synchronize()
    except Exception as exc:
        cap = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        n = int(torch.cuda.device_count())
        if n > 1:
            multi = (
                "\n  [bold]If you have a second GPU[/] (e.g. Ampere/Ada), pin this process to it:\n"
                "    CUDA_VISIBLE_DEVICES=1 python worker.py ...\n"
            )
        else:
            multi = (
                "\n  Install a PyTorch build that includes kernels for this GPU "
                "(see https://pytorch.org/get-started/locally/ ), or use a supported GPU.\n"
            )
        console.print(
            "[red]PyTorch cannot run CUDA kernels on the current default GPU.[/]\n"
            f"  cuda:0  name={name!r}  capability=sm_{cap[0]}{cap[1]}\n"
            "  Scene YOLO and LP YOLO require working torch CUDA.\n"
            f"{multi}"
            f"  [dim]{type(exc).__name__}: {exc}[/]"
        )
        raise SystemExit(2) from exc


def _log_visible_torch_cuda_device() -> None:
    """Log which GPU this process uses as ``torch.cuda:0`` (after ``CUDA_VISIBLE_DEVICES`` remap)."""
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        console.print("[cyan]CUDA[/] torch: not available (CPU)")
        return
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
    console.print(
        f"[cyan]CUDA[/] CUDA_VISIBLE_DEVICES={vis}  →  torch cuda:0 = {name!r}  "
        f"capability sm_{cap[0]}{cap[1]}"
    )


def _log_model_ready(title: str, detail: str) -> None:
    console.print(f"  [green]OK[/] [bold]{title}[/]  [dim]— {detail}[/]")
