#!/usr/bin/env python3
"""
Skeleton: wire DFINE person boxes into the portable peeing kit.

**Integrated runner (this repo):** from project root::

  uv run python Basil_Test/draw_dfine_peeing.py path/to/video.mp4

This file is the minimal hook pattern if you embed portable_peeing elsewhere.
Replace ``run_dfine_persons`` with your model. Run from ``portable_peeing/``::

  python examples/dfine_peeing_example.py /path/to/video.mp4 -o outputs/out.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.detection_contract import (
    merge_person_and_motorcycle_detections,
    prepare_scene_detections,
)
from models.types import Detection
from pipelines.peeing_pipeline import PeeingPipelineOptions, load_peeing_models, run_peeing_pipeline_video
from settings import PEEING_DETECTION_CONFIDENCE


def run_dfine_persons(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TODO: DFINE person head — (N,4) pixel xyxy + (N,) scores."""
    raise NotImplementedError("Plug in DFINE person inference")


def run_dfine_motorcycles(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TODO: DFINE motorcycle/motorbike classes — required every frame (empty N ok if none visible)."""
    raise NotImplementedError("Plug in DFINE motorcycle inference")


def make_scene_detector():
    def scene_detector(frame_index: int, frame_bgr: np.ndarray, timestamp_sec: float) -> list[Detection]:
        del frame_index, timestamp_sec
        p_boxes, p_scores = run_dfine_persons(frame_bgr)
        m_boxes, m_scores = run_dfine_motorcycles(frame_bgr)
        return prepare_scene_detections(
            merge_person_and_motorcycle_detections(p_boxes, p_scores, m_boxes, m_scores),
            min_confidence=PEEING_DETECTION_CONFIDENCE,
        )

    return scene_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="DFINE + portable peeing example")
    parser.add_argument("video")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    if not Path(args.video).is_file():
        raise SystemExit(f"Video not found: {args.video}")

    bundle = load_peeing_models()
    try:
        rec = run_peeing_pipeline_video(
            bundle,
            args.video,
            args.output,
            scene_detector=make_scene_detector(),
            per_video_times_init_sec=0.0,
            models_init_sec=bundle.init_sec,
            pipeline_options=PeeingPipelineOptions(),
        )
    finally:
        bundle.cleanup()

    if not rec.success:
        raise SystemExit(rec.error or "pipeline failed")
    print(f"Saved {args.output}  events={len(rec.events)}")


if __name__ == "__main__":
    main()
