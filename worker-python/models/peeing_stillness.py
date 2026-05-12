"""BBox stillness metrics for peeing temporal gating (no OpenCV dependency)."""

from __future__ import annotations

import math
from typing import Tuple


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return float(inter / union) if union > 0 else 0.0


def bbox_is_still(
    prev: Tuple[float, float, float, float],
    bbox: Tuple[float, float, float, float],
    *,
    min_iou: float,
    max_center_motion_norm: float,
    max_size_change: float,
) -> bool:
    """True when scene-YOLO person box is stable frame-to-frame (cheap motion gate)."""
    iou = iou_xyxy(prev, bbox)
    if iou < float(min_iou):
        return False
    ax1, ay1, ax2, ay2 = map(float, prev)
    bx1, by1, bx2, by2 = map(float, bbox)
    ah = max(1e-6, ay2 - ay1)
    acx = 0.5 * (ax1 + ax2)
    acy = 0.5 * (ay1 + ay2)
    bcx = 0.5 * (bx1 + bx2)
    bcy = 0.5 * (by1 + by2)
    motion_norm = float(math.hypot(bcx - acx, bcy - acy) / ah)
    if motion_norm > float(max_center_motion_norm):
        return False
    aw = max(1e-6, ax2 - ax1)
    area_prev = aw * ah
    bh = max(1e-6, by2 - by1)
    bw = max(1e-6, bx2 - bx1)
    area_curr = bw * bh
    size_change = abs(area_curr - area_prev) / max(area_prev, 1e-6)
    if size_change > float(max_size_change):
        return False
    return True
