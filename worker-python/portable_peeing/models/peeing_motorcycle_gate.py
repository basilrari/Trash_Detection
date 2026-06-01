# worker-python/models/peeing_motorcycle_gate.py
"""BBox-only heuristic: is this person detection likely riding a motorcycle?"""

from __future__ import annotations

from typing import Sequence


def intersection_area_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    return float(iw * ih)


def expand_bbox_xyxy(
    bbox: tuple[float, float, float, float],
    frac_w: float,
    frac_h: float,
) -> tuple[float, float, float, float]:
    """Scale width/height about the bbox center by ``1 + frac_*``."""
    x1, y1, x2, y2 = map(float, bbox)
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    nw = w * (1.0 + max(0.0, float(frac_w)))
    nh = h * (1.0 + max(0.0, float(frac_h)))
    hw, hh = 0.5 * nw, 0.5 * nh
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def lower_body_xyxy(
    bbox: tuple[float, float, float, float],
    lower_body_fraction: float,
) -> tuple[float, float, float, float]:
    """Bottom ``lower_body_fraction`` of the bbox by height (clamped)."""
    x1, y1, x2, y2 = map(float, bbox)
    h = max(1e-6, y2 - y1)
    frac = min(1.0, max(0.01, float(lower_body_fraction)))
    y_top = y2 - h * frac
    return (x1, y_top, x2, y2)


def bottom_center_xyxy(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = map(float, bbox)
    return (0.5 * (x1 + x2), y2)


def point_in_rect_xyxy(
    px: float, py: float, rect: tuple[float, float, float, float]
) -> bool:
    rx1, ry1, rx2, ry2 = map(float, rect)
    return rx1 <= px <= rx2 and ry1 <= py <= ry2


def person_seated_on_motorcycle(
    person_bbox: tuple[float, float, float, float],
    motorcycle_bboxes: Sequence[tuple[float, float, float, float]],
    *,
    expand_x: float,
    expand_y: float,
    lower_body_fraction: float,
    overlap_threshold: float,
) -> bool:
    """True if lower-body / feet cues match one expanded motorcycle bbox."""
    if not motorcycle_bboxes:
        return False
    lower = lower_body_xyxy(person_bbox, lower_body_fraction)
    la = max(0.0, lower[2] - lower[0]) * max(0.0, lower[3] - lower[1])
    if la <= 0.0:
        return False
    bcx, bcy = bottom_center_xyxy(person_bbox)
    thr = float(overlap_threshold)
    for mb in motorcycle_bboxes:
        exp = expand_bbox_xyxy(mb, expand_x, expand_y)
        ratio = intersection_area_xyxy(lower, exp) / la
        if ratio < thr:
            continue
        if not point_in_rect_xyxy(bcx, bcy, exp):
            continue
        return True
    return False
