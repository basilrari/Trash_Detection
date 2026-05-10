from pathlib import Path
from typing import List

import numpy as np
from ultralytics import YOLO

from models.base import LicensePlateDetector
from models.ultralytics_call_stats import UltralyticsCallStats
from models.types import FrameData, LicensePlate


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


class LpDetector(LicensePlateDetector):
    """License-plate YOLO via TensorRT ``.engine`` (Ultralytics ``YOLO`` wrapper)."""

    def __init__(self, *, engine_path: str | None = None, plate_class_id: int = 0):
        from settings import (
            LP_CONFIDENCE,
            LP_ENGINE_PATH,
            LP_TRT_BATCH_SIZE,
            LP_TRT_DYNAMIC,
            LP_TRT_IMAGE_SIZE,
        )

        self.plate_class_id = plate_class_id
        self._conf = float(LP_CONFIDENCE)
        self._imgsz = int(LP_TRT_IMAGE_SIZE)
        self._trt_batch = max(1, int(LP_TRT_BATCH_SIZE))
        self._dynamic = bool(LP_TRT_DYNAMIC)
        eng = _resolve_path(engine_path or LP_ENGINE_PATH)
        if not eng.is_file():
            raise FileNotFoundError(
                f"LP TensorRT engine not found: {eng}\n"
                "Set LP_ENGINE_PATH in settings.py."
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

    def detect_plates(self, frames: List[FrameData]) -> List[List[LicensePlate]]:
        if not frames:
            return []

        images = [f.image for f in frames]
        self._call_stats_in += len(images)

        B = self._trt_batch
        out_all: List[List[LicensePlate]] = []
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
                classes=[self.plate_class_id],
                conf=self._conf,
                imgsz=self._imgsz,
            )
            if not isinstance(results, list):
                results = list(results)
            parsed = self._to_license_plates(results)
            out_all.extend(parsed[:valid])
        return out_all

    def _to_license_plates(self, results) -> List[List[LicensePlate]]:
        all_plates: List[List[LicensePlate]] = []
        for result in results:
            frame_plates: List[LicensePlate] = []
            if result.boxes is None:
                all_plates.append(frame_plates)
                continue
            for box in result.boxes:
                conf = float(box.conf)
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                frame_plates.append(LicensePlate(bbox=(x1, y1, x2, y2), text="", confidence=conf))
            all_plates.append(frame_plates)
        return all_plates
