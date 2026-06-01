"""Scene detection filter and peeing overlay drawing (shared with full pipeline)."""

from __future__ import annotations

from typing import List, Sequence

import cv2
import numpy as np

from models.peeing_detector import PeeingState
from models.peeing_pose_viz import draw_pose_peeing_overlay
from models.types import Detection

# Match Ultralytics COCO names for YOLO ``classes=[0,2,3,5,7]`` (person + road vehicles).
VEHICLE_LABELS = ("car", "truck", "bus", "motorcycle", "motorbike", "vehicle")
PERSON_LABELS = ("person",)


def _is_scene_detection(d: Detection) -> bool:
    """YOLO is restricted to person + vehicles; keep this aligned with ``YoloDetector.classes``."""
    return d.label in PERSON_LABELS or d.label in VEHICLE_LABELS


def _filter_scene_detections(detections: Sequence[Detection]) -> List[Detection]:
    return [d for d in detections if _is_scene_detection(d)]


def _draw_peeing_overlay(
    frame: np.ndarray,
    state: PeeingState,
    *,
    draw_pose: bool = False,
    hand_groin_y_threshold: float = 0.1,
    min_visibility: float = 0.45,
) -> None:
    """Pose debug overlay and confirmed peeing boxes (every decoded frame when active)."""
    if draw_pose and state.pose_viz:
        draw_pose_peeing_overlay(
            frame,
            state.pose_viz,
            hand_groin_y_threshold=hand_groin_y_threshold,
            min_visibility=min_visibility,
        )
    red_bgr = (0, 0, 255)
    thick = 6
    for x1, y1, x2, y2 in state.mark_bboxes:
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(frame, p1, p2, red_bgr, thick)
