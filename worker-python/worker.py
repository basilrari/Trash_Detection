#!/usr/bin/env python3
"""
Local video pipeline: YOLO (stride gate by default) → RF-DETR trash → license plate → OCR → annotated MP4.

Examples (paths relative to worker-python/):

  python worker.py
  python worker.py inputs/clip.mp4
  python worker.py inputs/clip.mp4 -o outputs/custom.mp4

Gating (defaults in ``settings.py``):
  The gate decides how often YOLO runs (and thus LP/OCR, which need YOLO boxes). RF-DETR runs
  only on YOLO frames where a **person or vehicle** is detected (same confidence as scene
  activity). See ``core/yolo_stride_gate.py``.

  Edit ``GATE_MODE``, ``CHUNK_SECONDS``, and stride fields in ``settings.py``, or pass the
  ``--gate`` / ``--yolo-*`` flags below to override those values in memory for this run only.

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
        description=(
            "Run YOLO + license-plate detector + OCR on a video file.\n\n"
            "GATE_MODE:\n"
            "  yolo — Default: YOLO-only stride gate (coarse when idle, denser after people/vehicles). "
            "RF-DETR runs only on YOLO frames that have a person or vehicle. Tuning: "
            "YOLO_COARSE_STRIDE, YOLO_DENSE_STRIDE, YOLO_DENSE_IDLE_MISS_STREAK in settings.py "
            "(or override with flags below). See Readme.md § Gating.\n"
            "  off  — No stride gate: within each time chunk (CHUNK_SECONDS), YOLO on every frame; "
            "RF-DETR only on frames in that chunk that have a person or vehicle."
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
            "Override settings.GATE_MODE for this run only: "
            "'yolo' = coarse/dense YOLO stride gate; "
            "'off' = full YOLO within each time chunk (no stride gate). "
            "Default: value from settings.py."
        ),
    )
    parser.add_argument(
        "--yolo-coarse-stride",
        type=int,
        default=None,
        metavar="N",
        help=(
            "When GATE_MODE=yolo: run YOLO at least every N frames while no person/vehicle "
            "was seen on the last gated runs (typical 5–10). Overrides settings.YOLO_COARSE_STRIDE."
        ),
    )
    parser.add_argument(
        "--yolo-dense-stride",
        type=int,
        default=None,
        metavar="N",
        help=(
            "When GATE_MODE=yolo: after activity, run YOLO every N frames while the dense "
            "window is open (2 = every other frame). Overrides settings.YOLO_DENSE_STRIDE."
        ),
    )
    parser.add_argument(
        "--yolo-dense-idle-miss-streak",
        type=int,
        default=None,
        metavar="N",
        help=(
            "When GATE_MODE=yolo: exit dense sampling after N consecutive YOLO runs with no "
            "person/vehicle (default 10). Overrides settings.YOLO_DENSE_IDLE_MISS_STREAK."
        ),
    )

    args = parser.parse_args()

    import settings

    if args.gate is not None:
        settings.GATE_MODE = args.gate.strip().lower()
    if args.yolo_coarse_stride is not None:
        settings.YOLO_COARSE_STRIDE = int(args.yolo_coarse_stride)
    if args.yolo_dense_stride is not None:
        settings.YOLO_DENSE_STRIDE = int(args.yolo_dense_stride)
    if args.yolo_dense_idle_miss_streak is not None:
        settings.YOLO_DENSE_IDLE_MISS_STREAK = max(1, int(args.yolo_dense_idle_miss_streak))

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
