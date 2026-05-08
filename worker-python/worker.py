#!/usr/bin/env python3
"""
Local video pipeline: YOLO (stride gate by default) → RF-DETR trash → license plate → OCR → annotated MP4.

Examples (paths relative to worker-python/):

  python worker.py
  python worker.py inputs/clip.mp4
  python worker.py inputs/clip.mp4 -o outputs/custom.mp4

Gating (default ``GATE_MODE=yolo`` in ``settings.py``):
  The gate decides how often YOLO runs (and thus LP/OCR, which need YOLO boxes). RF-DETR
  runs on the same cadence as YOLO when the gate is on. See ``core/yolo_stride_gate.py``.

  # Full YOLO on every frame inside each time chunk (no stride gate)
  GATE_MODE=off python worker.py inputs/clip.mp4

  # Explicit coarse/dense tuning (defaults are already yolo in settings)
  GATE_MODE=yolo YOLO_COARSE_STRIDE=10 YOLO_DENSE_STRIDE=2 YOLO_DENSE_IDLE_MISS_STREAK=10 \\
    python worker.py inputs/clip.mp4 -o outputs/out.mp4

  # If PyTorch reports ``no kernel image`` on a very new GPU (e.g. Blackwell) but another GPU works:
  CUDA_VISIBLE_DEVICES=1 python worker.py inputs/clip.mp4 -o outputs/out.mp4
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def default_output_path(video_path: str, outputs_dir: str) -> str:
    p = Path(video_path)
    name = f"{p.stem}_annotated{p.suffix}"
    return str(Path(outputs_dir) / name)


def main() -> None:
    # -------------------------------------------------------------------------
    # Argparse must run before importing ``settings`` so ``os.environ`` overrides
    # (e.g. --gate yolo) are visible when defaults are read from the environment.
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO + license-plate detector + OCR on a video file.\n\n"
            "GATE_MODE:\n"
            "  yolo — Default: YOLO-only stride gate (coarse when idle, denser after people/vehicles). "
            "RF-DETR runs with the same schedule. Tuning: YOLO_COARSE_STRIDE, YOLO_DENSE_STRIDE, "
            "YOLO_DENSE_IDLE_MISS_STREAK (env or settings.py). See Readme.md § Gating.\n"
            "  off  — No stride gate: within each time chunk (CHUNK_SECONDS), YOLO on every frame; "
            "RF-DETR runs on every frame in each chunk."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Input video path (default: VIDEO_PATH in settings.py when omitted).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Annotated output path (default: OUTPUT_VIDEO in settings, or <stem>_annotated in outputs/).",
    )
    parser.add_argument(
        "--gate",
        choices=["off", "yolo"],
        default=None,
        help=(
            "Override GATE_MODE for this run only: "
            "'yolo' = coarse/dense YOLO stride gate (default in settings); "
            "'off' = full YOLO within each time chunk (no stride gate). "
            "Default: value from settings.py / environment."
        ),
    )
    parser.add_argument(
        "--yolo-coarse-stride",
        type=int,
        default=None,
        metavar="N",
        help=(
            "When GATE_MODE=yolo: run YOLO at least every N frames while no person/vehicle "
            "was seen on the last gated runs (typical 5–10). Env: YOLO_COARSE_STRIDE."
        ),
    )
    parser.add_argument(
        "--yolo-dense-stride",
        type=int,
        default=None,
        metavar="N",
        help=(
            "When GATE_MODE=yolo: after activity, run YOLO every N frames while the dense "
            "window is open (2 = every other frame). Env: YOLO_DENSE_STRIDE."
        ),
    )
    parser.add_argument(
        "--yolo-dense-idle-miss-streak",
        type=int,
        default=None,
        metavar="N",
        help=(
            "When GATE_MODE=yolo: exit dense sampling after N consecutive YOLO runs with no "
            "person/vehicle (default 10). Env: YOLO_DENSE_IDLE_MISS_STREAK."
        ),
    )

    args = parser.parse_args()

    if args.gate is not None:
        os.environ["GATE_MODE"] = args.gate
    if args.yolo_coarse_stride is not None:
        os.environ["YOLO_COARSE_STRIDE"] = str(args.yolo_coarse_stride)
    if args.yolo_dense_stride is not None:
        os.environ["YOLO_DENSE_STRIDE"] = str(args.yolo_dense_stride)
    if args.yolo_dense_idle_miss_streak is not None:
        os.environ["YOLO_DENSE_IDLE_MISS_STREAK"] = str(args.yolo_dense_idle_miss_streak)

    from settings import OUTPUT_VIDEO, OUTPUTS_DIR, VIDEO_PATH
    from pipelines.test_pipeline import run_pipeline

    video_path = args.video or VIDEO_PATH
    if not os.path.isfile(video_path):
        raise SystemExit(f"Video not found: {video_path}")

    out_path = args.output
    if out_path is None:
        out_path = OUTPUT_VIDEO if args.video is None else default_output_path(video_path, OUTPUTS_DIR)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    run_pipeline(video_path, out_path)


if __name__ == "__main__":
    main()
