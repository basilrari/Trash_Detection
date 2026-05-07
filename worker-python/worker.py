#!/usr/bin/env python3
"""
Local video pipeline entrypoint (YOLO → license plate → OCR → annotated video).

  python worker.py                    # uses VIDEO_PATH / OUTPUT_VIDEO in settings.py
  python worker.py clip.mp4           # input only; output name derived from input
  python worker.py clip.mp4 -o out.mp4
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from settings import OUTPUT_VIDEO, VIDEO_PATH


def default_output_path(video_path: str) -> str:
    p = Path(video_path)
    return str(p.with_name(f"{p.stem}_annotated{p.suffix}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YOLO + license-plate detector + OCR on a video file.",
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help=f"Input video path (default: VIDEO_PATH in settings.py, currently {VIDEO_PATH!r})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=f"Annotated output video (default: OUTPUT_VIDEO in settings or <name>_annotated.mp4)",
    )
    args = parser.parse_args()

    from pipelines.test_pipeline import run_pipeline

    video_path = args.video or VIDEO_PATH
    if not os.path.isfile(video_path):
        raise SystemExit(f"Video not found: {video_path}")

    out_path = args.output
    if out_path is None:
        out_path = OUTPUT_VIDEO if args.video is None else default_output_path(video_path)

    run_pipeline(video_path, out_path)


if __name__ == "__main__":
    main()
