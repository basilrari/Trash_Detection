"""Peeing overlay drawing. Scene boxes are prepared by the host detector (see ``detection_contract``)."""

from __future__ import annotations

import cv2
import numpy as np

from models.peeing_detector import PeeingState
from models.peeing_pose_viz import draw_pose_peeing_overlay


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
