#!/usr/bin/env python3
"""Print YOLO persons + MediaPipe pose dump for one video frame (tuning / copy-paste).

Run from anywhere; adds ``worker-python`` to ``sys.path``:

  python scripts/print_peeing_pose_debug.py path/to/video.mp4 --frame 1200
  python scripts/print_peeing_pose_debug.py path/to/video.mp4 --frame 0 --no-landmarks

Paste the JSON into a chat if you want thresholds or geometry rules adjusted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKER_ROOT = Path(__file__).resolve().parents[1]
if str(WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKER_ROOT))


def _strip_landmarks(obj: object) -> object:
    if isinstance(obj, dict):
        out: dict[str, object] = {}
        for k, v in obj.items():
            if k == "landmarks_normalized_in_crop" and isinstance(v, list):
                out[k] = f"<{len(v)} points omitted>"
            else:
                out[k] = _strip_landmarks(v)
        return out
    if isinstance(obj, list):
        return [_strip_landmarks(x) for x in obj]
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="0-based frame index to seek (default: 0)",
    )
    parser.add_argument(
        "--no-landmarks",
        action="store_true",
        help="Omit per-landmark x/y/z/visibility arrays (smaller paste)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent (default: 2; use 0 for one line)",
    )
    args = parser.parse_args()

    import cv2

    from core.types import FrameData
    from models.peeing_detector import PeeingDetector
    from models.yolo_detector import YoloDetector
    from settings import (
        PEEING_POSE_MODEL_PATH,
        PEEING_POSE_MODEL_URL,
        YOLO_CONFIDENCE,
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print("Could not open video:", args.video, file=sys.stderr)
        return 2

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(args.frame))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("Could not read frame", args.frame, file=sys.stderr)
        return 2

    h, w = frame.shape[:2]
    yolo = YoloDetector(conf_threshold=YOLO_CONFIDENCE)
    dets = yolo.detect([FrameData(index=args.frame, timestamp=args.frame / fps, image=frame)])[0]

    peeing = PeeingDetector(model_path=PEEING_POSE_MODEL_PATH, model_url=PEEING_POSE_MODEL_URL)
    try:
        reports = peeing.debug_person_reports(frame, dets, yolo_conf=YOLO_CONFIDENCE)
    finally:
        peeing.close()

    payload: dict[str, object] = {
        "video": str(args.video.resolve()),
        "frame_index": args.frame,
        "frame_size_wh": [w, h],
        "yolo_confidence_setting": YOLO_CONFIDENCE,
        "person_pose_reports": reports,
    }

    if args.no_landmarks:
        payload = _strip_landmarks(payload)  # type: ignore[assignment]

    indent = None if args.indent == 0 else args.indent
    print(json.dumps(payload, indent=indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
