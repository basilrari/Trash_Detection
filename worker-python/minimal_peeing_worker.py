#!/usr/bin/env python3
"""
Minimal peeing stack: **scene YOLO + PeeingDetector only** (same logic as ``peeing_worker.py``).

Does **not** import ``pipelines.test_pipeline`` (no RF-DETR, LP, or OCR). For details see
``pipelines/peeing_pipeline.py``.
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
        description="Minimal peeing: scene YOLO + PeeingDetector only (no trash/LP/OCR).",
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
    args = parser.parse_args()

    from settings import OUTPUTS_DIR, VIDEO_PATH
    from pipelines.peeing_pipeline import run_peeing_pipeline

    video_path = args.video or VIDEO_PATH
    if not os.path.isfile(video_path):
        raise SystemExit(f"Video not found: {video_path}")

    out_path = args.output or default_peeing_output_path(video_path, OUTPUTS_DIR)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    run_peeing_pipeline(video_path, out_path)


if __name__ == "__main__":
    main()
