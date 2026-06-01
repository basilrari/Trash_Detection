"""Draw YOLO pose skeletons and **video-pixel** hip→hand distances on full frames."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

# COCO-17 connectivity (0-indexed).
COCO17_SKELETON: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (9, 11),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
)

_LEFT_WRIST = 9
_RIGHT_WRIST = 10
_LEFT_HIP = 11
_RIGHT_HIP = 12


@dataclass(frozen=True)
class PosePersonViz:
    """Scene-YOLO person bbox + YOLO-pose keypoints mapped to full-frame pixels."""

    bbox: Tuple[float, float, float, float]
    keypoints_xy: Tuple[Tuple[float, float], ...]
    keypoints_conf: Tuple[float, ...]
    pose_hit: bool
    standing: bool
    min_wrist_groin_y: float
    still_ready: bool
    latched_confirm: bool
    consecutive_good_seconds: int
    hits_in_bucket: int
    standing_sides: str = ""


def _pt_ok(conf: Sequence[float], idx: int, min_vis: float) -> bool:
    return idx < len(conf) and float(conf[idx]) >= min_vis


def _hip_to_hand_video_px(
    kps: Sequence[Tuple[float, float]],
    conf: Sequence[float],
    min_vis: float,
) -> List[Tuple[str, float, Tuple[int, int], Tuple[int, int]]]:
    """
    Same-side hip → wrist distance in **frame pixels**, measured from pose keypoints
    on this frame (not from stored detector scores).
    """
    out: List[Tuple[str, float, Tuple[int, int], Tuple[int, int]]] = []
    for side, hi, wi in (("L", _LEFT_HIP, _LEFT_WRIST), ("R", _RIGHT_HIP, _RIGHT_WRIST)):
        if not (_pt_ok(conf, hi, min_vis) and _pt_ok(conf, wi, min_vis)):
            continue
        hx, hy = kps[hi]
        wx, wy = kps[wi]
        dist_px = math.hypot(wx - hx, wy - hy)
        hip_pt = (int(round(hx)), int(round(hy)))
        hand_pt = (int(round(wx)), int(round(wy)))
        out.append((side, dist_px, hip_pt, hand_pt))
    return out


def _draw_scene_yolo_person_bbox(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    *,
    color: Tuple[int, int, int],
    label: str,
    still_ready: bool,
) -> None:
    """Highlight scene-YOLO person box (pose runs on crop inside this rect)."""
    x1, y1, x2, y2 = bbox
    ix1, iy1 = int(round(x1)), int(round(y1))
    ix2, iy2 = int(round(x2)), int(round(y2))
    h, w = frame.shape[:2]
    ix1, iy1 = max(0, ix1), max(0, iy1)
    ix2, iy2 = min(w - 1, ix2), min(h - 1, iy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return

    overlay = frame.copy()
    fill = (color[0], color[1], color[2])
    cv2.rectangle(overlay, (ix1, iy1), (ix2, iy2), fill, -1)
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

    thick = 4
    cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, thick, cv2.LINE_AA)
    # Corner brackets (visible even on busy backgrounds)
    cw, ch = ix2 - ix1, iy2 - iy1
    tick = max(12, min(cw, ch) // 6)
    for (cx, cy, dx, dy) in (
        (ix1, iy1, 1, 1),
        (ix2, iy1, -1, 1),
        (ix1, iy2, 1, -1),
        (ix2, iy2, -1, -1),
    ):
        cv2.line(frame, (cx, cy), (cx + dx * tick, cy), color, thick, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * tick), color, thick, cv2.LINE_AA)

    still_tag = "still" if still_ready else "moving-box"
    tag = f"scene-YOLO person [{still_tag}]"
    _put_text_outline(
        frame,
        tag,
        (ix1 + 4, iy2 - 8),
        color=color,
        scale=0.55,
        thickness=2,
    )
    _put_text_outline(
        frame,
        label,
        (ix1 + 4, iy1 + 4),
        color=color,
        scale=0.5,
        thickness=1,
    )


def _put_text_outline(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    *,
    color: Tuple[int, int, int],
    scale: float = 0.55,
    thickness: int = 2,
) -> None:
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_pose_peeing_overlay(
    frame: np.ndarray,
    persons: Sequence[PosePersonViz],
    *,
    hand_groin_y_threshold: float,
    min_visibility: float = 0.45,
) -> None:
    """Skeleton + scene-YOLO box + hip→hand distance in video pixels."""
    if not persons:
        return
    thr = float(hand_groin_y_threshold)
    skel_color = (0, 220, 255)
    hip_color = (255, 200, 0)
    hand_line_color = (255, 0, 255)
    hit_color = (0, 255, 0)
    miss_color = (0, 0, 255)

    for pv in persons:
        x1, y1, x2, y2 = pv.bbox
        if pv.latched_confirm:
            box_color = (0, 0, 255)
            status = "CONFIRMED"
        elif pv.consecutive_good_seconds > 0 and pv.still_ready:
            box_color = (0, 165, 255)
            status = f"suspect {pv.consecutive_good_seconds}s"
        else:
            # Bright cyan so scene-YOLO box is easy to see (was thin gray).
            box_color = (255, 255, 0)
            status = "person"

        kps = pv.keypoints_xy
        conf = pv.keypoints_conf
        hip_hand = _hip_to_hand_video_px(kps, conf, min_visibility)

        if len(kps) >= 17:
            for a, b in COCO17_SKELETON:
                if _pt_ok(conf, a, min_visibility) and _pt_ok(conf, b, min_visibility):
                    pa = (int(round(kps[a][0])), int(round(kps[a][1])))
                    pb = (int(round(kps[b][0])), int(round(kps[b][1])))
                    cv2.line(frame, pa, pb, skel_color, 2, cv2.LINE_AA)

            for side, dist_px, hip_pt, hand_pt in hip_hand:
                wc = hit_color if pv.pose_hit else miss_color
                cv2.circle(frame, hip_pt, 6, hip_color, -1, cv2.LINE_AA)
                cv2.circle(frame, hand_pt, 6, wc, -1, cv2.LINE_AA)
                cv2.line(frame, hip_pt, hand_pt, hand_line_color, 3, cv2.LINE_AA)
                mx = (hip_pt[0] + hand_pt[0]) // 2
                my = (hip_pt[1] + hand_pt[1]) // 2
                label = f"{int(round(dist_px))} px"
                _put_text_outline(
                    frame,
                    label,
                    (mx + 6, my - 10),
                    color=(255, 255, 255),
                    scale=0.65,
                    thickness=2,
                )

        # Scene YOLO person bbox on top (stillness is measured on this box).
        _draw_scene_yolo_person_bbox(
            frame,
            pv.bbox,
            color=box_color,
            label=status,
            still_ready=pv.still_ready,
        )

        # Top banner: video measurement from keypoints on this frame.
        header_lines: List[str] = []
        header_lines.append("scene YOLO person  →  YOLO pose on crop")
        if hip_hand:
            parts = [f"{side} hip→hand {int(round(d))} px" for side, d, _, _ in hip_hand]
            header_lines.append("VIDEO distance: " + "   ".join(parts))
        else:
            header_lines.append("VIDEO distance: hip/hand not visible in pose")

        stand_s = pv.standing_sides if pv.standing_sides else "-"
        header_lines.append(
            f"{status}  standing={'yes' if pv.standing else 'no'} "
            f"(sides={stand_s})  rule_dY={pv.min_wrist_groin_y:.3f}  "
            f"still={'yes' if pv.still_ready else 'no'}  hits={pv.hits_in_bucket}"
        )

        line_h = 24
        ty0 = max(12, int(round(y1)) - 10 - line_h * len(header_lines))
        for i, line in enumerate(header_lines):
            ty = ty0 + i * line_h
            col = (255, 255, 0) if i == 1 else box_color if i == 0 else (180, 180, 180)
            scale = 0.58 if i == 1 else 0.48
            _put_text_outline(
                frame,
                line,
                (int(round(x1)), ty),
                color=col,
                scale=scale,
                thickness=2 if i == 1 else 1,
            )
