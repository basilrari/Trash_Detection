from pathlib import Path
from typing import List

import numpy as np
from ultralytics import YOLO

from models.base import Detector
from models.ultralytics_call_stats import UltralyticsCallStats
from models.types import FrameData, Detection


def _worker_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(p: str | Path) -> Path:
    pp = Path(p)
    if pp.is_file():
        return pp
    cand = _worker_root() / p
    if cand.is_file():
        return cand
    return pp


class YoloDetector(Detector):
    """Scene YOLO via TensorRT ``.engine`` (Ultralytics ``YOLO`` wrapper)."""

    def __init__(self, *, engine_path: str | None = None, conf_threshold: float = 0.5):
        from settings import (
            YOLO_ENGINE_PATH,
            YOLO_TRT_BATCH_SIZE,
            YOLO_TRT_DYNAMIC,
            YOLO_TRT_IMAGE_SIZE,
        )

        self.conf_threshold = conf_threshold
        self.classes = [0, 2, 3, 5, 7]
        self._imgsz = int(YOLO_TRT_IMAGE_SIZE)
        self._trt_batch = max(1, int(YOLO_TRT_BATCH_SIZE))
        self._dynamic = bool(YOLO_TRT_DYNAMIC)
        eng = _resolve_path(engine_path or YOLO_ENGINE_PATH)
        if not eng.is_file():
            raise FileNotFoundError(
                f"Scene YOLO TensorRT engine not found: {eng}\n"
                "Set YOLO_ENGINE_PATH in settings.py."
            )
        self.model = YOLO(str(eng))

        self._call_stats_in = 0
        self._call_stats_launches = 0
        self._call_stats_padded = 0
        self._call_stats_slack = 0

    def reset_inference_batch_stats(self) -> None:
        self._call_stats_in = 0
        self._call_stats_launches = 0
        self._call_stats_padded = 0
        self._call_stats_slack = 0

    def get_inference_batch_stats(self) -> UltralyticsCallStats:
        return UltralyticsCallStats(
            self._call_stats_in,
            self._call_stats_launches,
            self._call_stats_padded,
            self._call_stats_slack,
        )

    def detect(self, frames: List[FrameData]) -> List[List[Detection]]:
        if not frames:
            return []

        images = [f.image for f in frames]
        self._call_stats_in += len(images)

        B = self._trt_batch
        out_all: List[List[Detection]] = []
        for start in range(0, len(images), B):
            chunk = images[start : start + B]
            valid = len(chunk)
            self._call_stats_launches += 1
            short = max(0, B - valid)
            if self._dynamic:
                self._call_stats_slack += short
            else:
                self._call_stats_padded += short
            if not self._dynamic and valid < B:
                blank = np.zeros_like(chunk[0])
                chunk = list(chunk) + [blank] * (B - valid)
            results = self.model(
                chunk,
                classes=self.classes,
                conf=self.conf_threshold,
                imgsz=self._imgsz,
            )
            if not isinstance(results, list):
                results = list(results)
            parsed = self._results_to_detections(results)
            out_all.extend(parsed[:valid])
        return out_all

    @staticmethod
    def _results_to_detections(results) -> List[List[Detection]]:
        detections_per_frame: List[List[Detection]] = []
        for result in results:
            frame_dets: List[Detection] = []
            names = result.names
            if result.boxes is None:
                detections_per_frame.append(frame_dets)
                continue
            for box in result.boxes:
                cls_id = int(box.cls)
                label = names.get(cls_id, str(cls_id))
                conf = float(box.conf)
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                frame_dets.append(Detection(bbox=(x1, y1, x2, y2), label=label, confidence=conf))
            detections_per_frame.append(frame_dets)
        return detections_per_frame
