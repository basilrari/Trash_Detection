#!/usr/bin/env python3
"""
Local video pipeline: YOLO (stride gate by default) → RF-DETR trash → license plate → OCR → annotated MP4.

Examples (paths relative to worker-python/):

  python worker.py
  python worker.py inputs/clip.mp4
  python worker.py inputs/clip.mp4 -o outputs/custom.mp4

Gating (``settings.py`` only — no CLI overrides for gate or YOLO strides):
  ``GATE_MODE`` may be ``off``, ``yolo``, or a positive integer string (``"1"``, ``"2"``, …).
  Integer mode: scene YOLO only on every Nth frame, micro-batched (see ``pipelines.test_pipeline``).
  ``yolo`` uses ``core/yolo_stride_gate.py`` (coarse/dense). RF-DETR still follows scene activity
  on frames where YOLO ran (integer mode: sampled frames only).

  If PyTorch reports ``no kernel image`` on a very new GPU (e.g. Blackwell) but another GPU works,
  launch with a different visible GPU at the shell (e.g. ``CUDA_VISIBLE_DEVICES=1``) before
  ``python worker.py`` — that is an OS / driver choice, not read from app ``settings.py``.
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
    parser = argparse.ArgumentParser(
        description="Run YOLO + license-plate detector + OCR on a video file (gating: settings.py only).",
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

    args = parser.parse_args()

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
