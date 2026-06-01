#!/usr/bin/env python3
"""
Peeing-only video pipeline: scene YOLO + PeeingDetector → annotated MP4 (no trash, LP, or OCR).

Run from ``worker-python/``::

  python peeing_worker.py
  python peeing_worker.py inputs/clip.mp4
  python peeing_worker.py inputs/clip.mp4 -o outputs/clip_peeing_annotated.mp4

Stride and thresholds come from ``settings.py`` only (same as ``worker.py``).
Default output when you pass an input path is ``outputs/<stem>_peeing_annotated.<ext>``.
When you omit the video argument, ``VIDEO_PATH`` from settings is used and the default output
is also ``outputs/<stem>_peeing_annotated.<ext>`` (so the full-pipeline ``OUTPUT_VIDEO`` path is not overwritten).

Equivalent CLI: ``minimal_peeing_worker.py`` (same ``main()``).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def default_peeing_output_path(video_path: str, outputs_dir: str) -> str:
    p = Path(video_path)
    name = f"{p.stem}_peeing_annotated{p.suffix}"
    return str(Path(outputs_dir) / name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run scene YOLO + peeing detection only (no RF-DETR, LP, or OCR).",
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
        help="Annotated output path (default: outputs/<stem>_peeing_annotated.<ext>).",
    )
    parser.add_argument(
        "--hand-groin-y",
        type=float,
        default=None,
        metavar="THR",
        help="Override PEEING_HAND_GROIN_Y_THRESHOLD (normalized crop Y).",
    )
    parser.add_argument(
        "--pose-viz",
        action="store_true",
        help="Draw YOLO pose skeleton + groin-distance debug on output frames.",
    )

    args = parser.parse_args()

    from settings import OUTPUTS_DIR, VIDEO_PATH
    from pipelines.peeing_pipeline import PeeingPipelineOptions, run_peeing_pipeline

    video_path = args.video or VIDEO_PATH
    if not os.path.isfile(video_path):
        raise SystemExit(f"Video not found: {video_path}")

    out_path = args.output or default_peeing_output_path(video_path, OUTPUTS_DIR)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    opts = PeeingPipelineOptions(
        hand_groin_y_threshold=args.hand_groin_y,
        draw_pose=args.pose_viz,
        collect_pose_viz=args.pose_viz,
    )
    run_peeing_pipeline(video_path, out_path, options=opts)


if __name__ == "__main__":
    main()
