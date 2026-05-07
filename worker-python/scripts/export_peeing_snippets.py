#!/usr/bin/env python3
"""Save PNG/JPG crops from video (debug export).

**``--score-mode production`` (default)** — same ``PeeingDetector`` kwargs and ``update()``
as the worker. On every frame where **YOLO runs** (``run_yolo=True`` with
``--gate-mode yolo``, or every frame with ``--gate-mode off``), writes **one chip per
person** with pose overlay and **only** a **percent**: by default
``max(production instant, cross-arm / groin, bend+groin)``; experiment terms are calibrated
so a **bent torso + one or two hands at the fly** (your baseline) reads near **100%** when
landmarks are confident. A **pose guard** (wrist height vs shoulders, arm split, nose
proximity) scales the chip down for hat/salute/wave false positives. Unless
``--no-experiment-cross-arm`` is set (production instant only; guard still applies).
Saves **all** such frames
in the window (no score cutoff) when MediaPipe returns landmarks. ``--instant-hit-range``
limits frame indices; production still warms up from frame **0**.

**``--instant-hit-range FIRST LAST``** — save only frames FIRST..LAST (inclusive).
**Export-debug** seeks to FIRST (no warmup). Use ``-o`` for the output directory.

**``--score-mode export-debug``** — legacy: every frame, export-only score; uses
``--threshold`` to filter; does not call ``update()``.

Example::

  cd worker-python
  python scripts/export_peeing_snippets.py inputs/clip.mp4 -o outputs/peeing_snips
  python scripts/export_peeing_snippets.py inputs/clip.mp4 -o out --score-mode export-debug
  python scripts/export_peeing_snippets.py inputs/clip.mp4 -o out --instant-hit-range 600 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

WORKER_ROOT = Path(__file__).resolve().parents[1]
if str(WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKER_ROOT))

from core.types import Detection, FrameData  # noqa: E402
from models.peeing_detector import PeeingDetector  # noqa: E402
from models.yolo_detector import YoloDetector  # noqa: E402
from pipelines.test_pipeline import _filter_scene_detections, _scene_has_activity  # noqa: E402
from settings import (  # noqa: E402
    GATE_MODE,
    PEEING_ALARM_ENTER_HIT_FRACTION,
    PEEING_ALARM_EXIT_HIT_FRACTION,
    PEEING_ALARM_MIN_SAMPLES,
    PEEING_CROP_MARGIN,
    PEEING_GROIN_DIST_MAX,
    PEEING_GROIN_LOOSE_FACTOR,
    PEEING_MIN_VISIBILITY,
    PEEING_PELVIC_BAND_Y_ABOVE,
    PEEING_PELVIC_BAND_Y_BELOW,
    PEEING_POSE_MATCH_THRESHOLD,
    PEEING_POSE_MODEL_PATH,
    PEEING_POSE_MODEL_URL,
    PEEING_POSE_STRIDE,
    PEEING_SQUAT_DEPTH_SCALE,
    PEEING_SQUAT_HIP_KNEE_GAP_MAX,
    PEEING_STANDING_Y_MARGIN,
    PEEING_WINDOW_SEC,
    PEEING_WRIST_BAND_MIN_VISIBILITY,
    YOLO_COARSE_STRIDE,
    YOLO_CONFIDENCE,
    YOLO_DENSE_IDLE_MISS_STREAK,
    YOLO_DENSE_STRIDE,
)


# --- Export-only instant score (edit here; production stays in peeing_detector) ---


def _export_debug_upright_standing(peeing: PeeingDetector, lms: list, PL) -> bool:
    lh = lms[PL.LEFT_HIP.value]
    rh = lms[PL.RIGHT_HIP.value]
    lk = lms[PL.LEFT_KNEE.value]
    rk = lms[PL.RIGHT_KNEE.value]

    def vis_ok(lm: object) -> bool:
        return peeing._lm_vis(lm) >= peeing.min_visibility

    if (
        vis_ok(lh)
        and vis_ok(rh)
        and vis_ok(lk)
        and vis_ok(rk)
        and lh.y is not None
        and rh.y is not None
        and lk.y is not None
        and rk.y is not None
    ):
        left_ok = float(lh.y) + peeing.standing_y_margin < float(lk.y)
        right_ok = float(rh.y) + peeing.standing_y_margin < float(rk.y)
        strong = (
            min(peeing._lm_vis(lh), peeing._lm_vis(rh), peeing._lm_vis(lk), peeing._lm_vis(rk))
            >= 0.38
        )
        if strong:
            return left_ok and right_ok
        hip_m = (float(lh.y) + float(rh.y)) * 0.5
        knee_m = (float(lk.y) + float(rk.y)) * 0.5
        return hip_m + 0.05 < knee_m

    leg_ok = False
    if vis_ok(lh) and vis_ok(lk) and lh.y is not None and lk.y is not None:
        if lh.y + peeing.standing_y_margin < lk.y:
            leg_ok = True
    if vis_ok(rh) and vis_ok(rk) and rh.y is not None and rk.y is not None:
        if rh.y + peeing.standing_y_margin < rk.y:
            leg_ok = True
    return leg_ok


def _export_debug_seated_suppressor(peeing: PeeingDetector, lms: list) -> float:
    """Crush standing-like score for folded thighs (motorcycle / seated)."""
    PL = peeing._PoseLandmark
    lh, rh = lms[PL.LEFT_HIP.value], lms[PL.RIGHT_HIP.value]
    lk, rk = lms[PL.LEFT_KNEE.value], lms[PL.RIGHT_KNEE.value]
    la, ra = lms[PL.LEFT_ANKLE.value], lms[PL.RIGHT_ANKLE.value]

    def vis_y(lm: object, t: float) -> bool:
        return peeing._lm_vis(lm) >= t and lm.y is not None

    if not (
        vis_y(lh, 0.34)
        and vis_y(rh, 0.34)
        and vis_y(lk, 0.34)
        and vis_y(rk, 0.34)
    ):
        return 1.0

    hip_m = (float(lh.y) + float(rh.y)) * 0.5
    knee_m = (float(lk.y) + float(rk.y)) * 0.5
    thigh_y = knee_m - hip_m

    if vis_y(la, 0.28) and vis_y(ra, 0.28):
        ankle_m = (float(la.y) + float(ra.y)) * 0.5
        calf_y = ankle_m - knee_m
        if thigh_y < 0.092 and calf_y > 0.052:
            return 0.06
        if thigh_y < 0.068:
            return 0.12
    else:
        if thigh_y < 0.072:
            return 0.18

    return 1.0


def _export_wrist_hip_overlap_gate(peeing: PeeingDetector, lms: list) -> float:
    """Scale standing cue: true peeing has wrists almost on hip points; far = not peeing.

    Uses Euclidean distance in **normalized crop** space. Tune ``tight`` /
    ``loose`` in this function while iterating on exports.
    """
    PL = peeing._PoseLandmark
    lw = lms[PL.LEFT_WRIST.value]
    rw = lms[PL.RIGHT_WRIST.value]
    lh = lms[PL.LEFT_HIP.value]
    rh = lms[PL.RIGHT_HIP.value]

    tight = 0.048
    loose = 0.135

    def dist2(a: object, b: object) -> float:
        if a.x is None or a.y is None or b.x is None or b.y is None:
            return 9.0
        return float(np.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y)))

    def wrist_near_hip(wrist: object) -> float:
        if peeing._lm_vis(wrist) < 0.32:
            return 0.35
        d = min(dist2(wrist, lh), dist2(wrist, rh))
        if d <= tight:
            return 1.0
        if d >= loose:
            return 0.0
        return float(np.clip((loose - d) / max(loose - tight, 1e-6), 0.0, 1.0))

    vl = peeing._lm_vis(lw) >= 0.32
    vr = peeing._lm_vis(rw) >= 0.32
    ql = wrist_near_hip(lw)
    qr = wrist_near_hip(rw)
    if vl and vr:
        return float(max(0.0, min(1.0, ql * qr)))
    if vl:
        return float(max(0.0, min(1.0, ql)))
    if vr:
        return float(max(0.0, min(1.0, qr)))
    return 0.0


def _export_debug_standing_pee_score(peeing: PeeingDetector, lms: list) -> float:
    """Standing branch mirroring ``PeeingDetector._standing_pee_score`` with export tweaks."""
    PL = peeing._PoseLandmark

    def lm_at(idx: int):
        return lms[idx]

    if not _export_debug_upright_standing(peeing, lms, PL):
        return 0.0

    lh = lm_at(PL.LEFT_HIP.value)
    rh = lm_at(PL.RIGHT_HIP.value)
    lk = lm_at(PL.LEFT_KNEE.value)
    rk = lm_at(PL.RIGHT_KNEE.value)
    lw = lm_at(PL.LEFT_WRIST.value)
    rw = lm_at(PL.RIGHT_WRIST.value)
    ls = lm_at(PL.LEFT_SHOULDER.value)
    rs = lm_at(PL.RIGHT_SHOULDER.value)

    def vis_ok(lm: object) -> bool:
        return peeing._lm_vis(lm) >= peeing.min_visibility

    groin_x = groin_y = None
    if vis_ok(lh) and vis_ok(rh) and lh.x is not None and rh.x is not None:
        groin_x = (lh.x + rh.x) * 0.5
        groin_y = (lh.y + rh.y) * 0.5  # type: ignore[operator]
    elif vis_ok(lh) and lh.x is not None and lh.y is not None:
        groin_x, groin_y = lh.x, lh.y
    elif vis_ok(rh) and rh.x is not None and rh.y is not None:
        groin_x, groin_y = rh.x, rh.y

    if groin_x is None or groin_y is None:
        return 0.0

    gwx, gwy = float(groin_x), float(groin_y)
    tight = max(peeing.groin_dist_max, 1e-6)
    loose = tight * peeing.groin_loose_factor

    def wrist_prox_score(wrist) -> float:
        if not vis_ok(wrist) or wrist.x is None or wrist.y is None:
            return 0.0
        wx, wy = float(wrist.x), float(wrist.y)
        dists = [float(np.hypot(wx - gwx, wy - gwy))]
        if vis_ok(lh) and lh.x is not None and lh.y is not None:
            dists.append(float(np.hypot(wx - float(lh.x), wy - float(lh.y))))
        if vis_ok(rh) and rh.x is not None and rh.y is not None:
            dists.append(float(np.hypot(wx - float(rh.x), wy - float(rh.y))))
        d = min(dists)
        if d <= tight:
            return float(min(1.0, 1.0 - d / tight))
        if d <= loose:
            span = loose - tight
            return float(0.52 * min(1.0, (loose - d) / max(span, 1e-6)))
        return 0.0

    best_prox = max(wrist_prox_score(lw), wrist_prox_score(rw))

    band_score = 0.0
    if (
        vis_ok(ls)
        and vis_ok(rs)
        and vis_ok(lh)
        and vis_ok(rh)
        and ls.x is not None
        and rs.x is not None
        and ls.y is not None
        and rs.y is not None
        and lh.x is not None
        and rh.x is not None
    ):
        smy = (float(ls.y) + float(rs.y)) * 0.5
        body_w = max(
            abs(float(lh.x) - float(rh.x)),
            abs(float(ls.x) - float(rs.x)),
            0.07,
        ) * 1.12
        for wrist in (lw, rw):
            if peeing._lm_vis(wrist) < peeing.wrist_band_min_visibility:
                continue
            if wrist.x is None or wrist.y is None:
                continue
            wx, wy = float(wrist.x), float(wrist.y)
            if wy < smy - 0.02:
                continue
            if wy < gwy + peeing.pelvic_band_y_above:
                continue
            if wy > gwy + peeing.pelvic_band_y_below:
                continue
            if abs(wx - gwx) > body_w * 0.74:
                continue
            band_score = max(band_score, 0.88)

    eff = max(best_prox, band_score * 0.88)
    if eff <= 0.0:
        return 0.0
    base = float(min(1.0, 0.34 + 0.66 * eff))
    hip_gate = _export_wrist_hip_overlap_gate(peeing, lms)
    return float(min(1.0, base * hip_gate))


def _export_debug_instant_score(peeing: PeeingDetector, lms: list) -> float:
    st = _export_debug_standing_pee_score(peeing, lms) * _export_debug_seated_suppressor(peeing, lms)
    sq = peeing._squat_score(lms)
    sm = peeing._straddle_penalty(lms)
    return float(min(1.0, max(0.0, max(st, sq) * sm)))


# --- Export-only experiment (chip label only; does not affect PeeingDetector.update) ---
# Full-strength blend so a strong cue can read as 100% on the chip (not capped ~90%).
_EXPERIMENT_CROSS_ARM_WEIGHT = 1.0
_EXPERIMENT_BEND_GROIN_WEIGHT = 1.0


def _export_effective_torso_width(peeing: PeeingDetector, lms: list, PL) -> float:
    """Lateral scale in normalized crop coords. Hip span collapses in profile; use shoulders."""
    lh, rh = lms[PL.LEFT_HIP.value], lms[PL.RIGHT_HIP.value]
    ls, rs = lms[PL.LEFT_SHOULDER.value], lms[PL.RIGHT_SHOULDER.value]

    def gv(lm: object, t: float = 0.28) -> bool:
        return peeing._lm_vis(lm) >= t and lm.x is not None and lm.y is not None

    hip_w = abs(float(lh.x) - float(rh.x)) if gv(lh) and gv(rh) else 0.0
    shoulder_w = abs(float(ls.x) - float(rs.x)) if gv(ls) and gv(rs) else 0.0
    return float(max(hip_w, shoulder_w * 0.54, 0.104))


def _export_experimental_cross_wrist_opposite_hip(peeing: PeeingDetector, lms: list) -> float:
    """0..1: diagonal wrist–hip reach, or wrist(s) at groin (one or two hands).

    Tuned so a bent torso + hand(s) at the fly reads near **100%** on the chip (export only).
    """
    PL = peeing._PoseLandmark
    lw, rw = lms[PL.LEFT_WRIST.value], lms[PL.RIGHT_WRIST.value]
    lh, rh = lms[PL.LEFT_HIP.value], lms[PL.RIGHT_HIP.value]
    ls, rs = lms[PL.LEFT_SHOULDER.value], lms[PL.RIGHT_SHOULDER.value]

    def gv(lm: object, t: float = 0.28) -> bool:
        return peeing._lm_vis(lm) >= t and lm.x is not None and lm.y is not None

    if not all(gv(x) for x in (lw, rw, lh, rh)):
        return 0.0

    eff_w = _export_effective_torso_width(peeing, lms, PL)
    gwx = (float(lh.x) + float(rh.x)) * 0.5
    gwy = (float(lh.y) + float(rh.y)) * 0.5

    # Left wrist → right hip / right wrist → left hip (diagonal across pelvis)
    d_lr = float(np.hypot(float(lw.x) - float(rh.x), float(lw.y) - float(rh.y)))
    d_rl = float(np.hypot(float(rw.x) - float(lh.x), float(rw.y) - float(lh.y)))

    tight = 0.44 * eff_w + 0.036
    loose = tight * 2.15

    def falloff(d: float) -> float:
        if d <= tight:
            return 1.0 - d / max(tight, 1e-6)
        if d <= loose:
            span = loose - tight
            return 0.62 * (1.0 - (d - tight) / max(span, 1e-6))
        return 0.0

    diag = max(falloff(d_lr), falloff(d_rl), 0.0)

    # Groin line: two-hand cluster or single best wrist (profile / one-hand-at-fly)
    sg = 0.92 * eff_w + 0.078
    dl_g = float(np.hypot(float(lw.x) - gwx, float(lw.y) - gwy))
    dr_g = float(np.hypot(float(rw.x) - gwx, float(rw.y) - gwy))
    twin = max(0.0, 1.0 - min(dl_g, dr_g) / max(sg, 1e-6))
    solo = max(
        max(0.0, 1.0 - dl_g / max(sg, 1e-6)),
        max(0.0, 1.0 - dr_g / max(sg, 1e-6)),
    )
    if gv(ls) and gv(rs):
        smy_c = (float(ls.y) + float(rs.y)) * 0.5
        hmy_c = (float(lh.y) + float(rh.y)) * 0.5
        th = max(hmy_c - smy_c, 0.052)
        spread_y = abs(float(lw.y) - float(rw.y)) / max(th, 1e-6)
        if spread_y > 0.33:
            solo = min(solo, twin + 0.12, 0.42)
    inner = max(twin, 0.92 * solo)
    outer = max(0.0, 1.0 - max(dl_g, dr_g) / max(1.48 * eff_w + 0.12, 1e-6))
    groin_pair = float(min(1.0, inner * (0.38 + 0.62 * outer)))

    if gv(ls) and gv(rs):
        smy = (float(ls.y) + float(rs.y)) * 0.5
        wy = min(float(lw.y), float(rw.y))
        if wy < smy - 0.11:
            groin_pair *= 0.72

    return float(min(1.0, max(diag, groin_pair)))


def _export_experimental_bend_groin_score(peeing: PeeingDetector, lms: list) -> float:
    """0..1: bent torso + wrist(s) at groin — **baseline** pose targets ~100% (export only).

    Accepts **one or two** hands: per-wrist groin distance and relaxed ``together`` when
    ``near`` and bend agree (other arm can reach across the body).
    """
    PL = peeing._PoseLandmark
    lw, rw = lms[PL.LEFT_WRIST.value], lms[PL.RIGHT_WRIST.value]
    lh, rh = lms[PL.LEFT_HIP.value], lms[PL.RIGHT_HIP.value]
    ls, rs = lms[PL.LEFT_SHOULDER.value], lms[PL.RIGHT_SHOULDER.value]

    def gv(lm: object, t: float = 0.28) -> bool:
        return peeing._lm_vis(lm) >= t and lm.x is not None and lm.y is not None

    if not all(gv(x) for x in (lw, rw, lh, rh, ls, rs)):
        return 0.0

    eff_w = _export_effective_torso_width(peeing, lms, PL)
    gwx = (float(lh.x) + float(rh.x)) * 0.5
    gwy = (float(lh.y) + float(rh.y)) * 0.5
    smy = (float(ls.y) + float(rs.y)) * 0.5
    hmy = (float(lh.y) + float(rh.y)) * 0.5
    torso_h = max(hmy - smy, 0.052)

    # Slightly generous scale so real fly-zone poses land high (calibrated to user baseline).
    scale = max(0.92 * eff_w + 0.074, 1e-6)
    dl = float(np.hypot(float(lw.x) - gwx, float(lw.y) - gwy))
    dr = float(np.hypot(float(rw.x) - gwx, float(rw.y) - gwy))
    twin = max(0.0, 1.0 - min(dl, dr) / scale)
    solo = max(max(0.0, 1.0 - dl / scale), max(0.0, 1.0 - dr / scale))
    spread_y = abs(float(lw.y) - float(rw.y)) / max(torso_h, 1e-6)
    if spread_y > 0.33:
        solo = min(solo, twin + 0.12, 0.42)
    near = float(min(1.0, max(twin, 0.94 * solo)))

    w_sep = float(np.hypot(float(lw.x) - float(rw.x), float(lw.y) - float(rw.y)))
    pair = max(0.0, 1.0 - w_sep / max(1.02 * eff_w + 0.118, 1e-6))

    gap = hmy - smy
    gap_bend = 0.0
    if gap < 0.24:
        gap_bend = min(1.0, (0.24 - gap) / 0.118)

    wmy = (float(lw.y) + float(rw.y)) * 0.5
    band = max(0.15 * torso_h + 0.048, 0.056)
    depth_align = max(0.0, 1.0 - abs(wmy - hmy) / max(band, 1e-6))
    bend = max(gap_bend, depth_align * 0.97)

    # Two hands close, or one hand at groin while bent (other wrist may sit farther).
    together = max(pair, 0.93 * near * (0.12 + 0.88 * max(0.35, min(1.0, 1.05 * bend))))

    wy = min(float(lw.y), float(rw.y))
    low = 1.0 if wy >= smy - 0.09 else 0.86

    mix = 0.34 * near + 0.32 * together + 0.36 * bend
    mn = min(near, together, bend)
    mx = max(near, together, bend)
    raw = min(1.0, max(mx, mix * 1.18 - 0.06, 0.72 * mx + 0.28 * mix + 0.14 * mn)) * low
    return float(min(1.0, max(0.0, raw)))


def _export_chip_pose_guard(peeing: PeeingDetector, lms: list) -> float:
    """0..1 multiplier for the chip percent — kills hat/salute/wave false positives.

    ``max(prod, cross, bend)`` can sit near 1 when **one** wrist hangs near the groin in 2D
    while the other is at the head; production standing geometry can do the same. This
    guard requires plausible **working-zone** wrists (export only).
    """
    PL = peeing._PoseLandmark
    lw, rw = lms[PL.LEFT_WRIST.value], lms[PL.RIGHT_WRIST.value]
    ls, rs = lms[PL.LEFT_SHOULDER.value], lms[PL.RIGHT_SHOULDER.value]
    lh, rh = lms[PL.LEFT_HIP.value], lms[PL.RIGHT_HIP.value]

    def gv(lm: object, t: float = 0.24) -> bool:
        return peeing._lm_vis(lm) >= t and lm.x is not None and lm.y is not None

    if not all(gv(x) for x in (lw, rw, ls, rs, lh, rh)):
        return 1.0

    smy = (float(ls.y) + float(rs.y)) * 0.5
    hmy = (float(lh.y) + float(rh.y)) * 0.5
    torso_h = max(hmy - smy, 0.05)

    # Smaller y = higher on screen. ``hi`` = topmost wrist.
    hi = min(float(lw.y), float(rw.y))
    if hi >= smy + 0.018:
        upper = 1.0
    elif hi >= smy - 0.058:
        u = (hi - (smy - 0.058)) / max(0.074, 1e-6)
        u = max(0.0, min(1.0, u))
        upper = 0.18 + 0.82 * u
    else:
        u = (hi - (smy - 0.132)) / max(0.078, 1e-6)
        u = max(0.0, min(1.0, u))
        upper = max(0.05, 0.2 * u)

    spread = abs(float(lw.y) - float(rw.y)) / max(torso_h, 1e-6)
    if spread <= 0.34:
        split = 1.0
    else:
        split = max(0.1, 1.0 - (spread - 0.34) * 2.35)

    nose_pen = 1.0
    try:
        nose_idx = peeing._PoseLandmark.NOSE.value
        nose_lm = lms[nose_idx]
    except (AttributeError, IndexError):
        nose_lm = None
    if nose_lm is not None and gv(nose_lm, 0.22):
        eff_w = _export_effective_torso_width(peeing, lms, PL)
        nx, ny = float(nose_lm.x), float(nose_lm.y)
        nd = min(
            float(np.hypot(float(lw.x) - nx, float(lw.y) - ny)),
            float(np.hypot(float(rw.x) - nx, float(rw.y) - ny)),
        )
        thr = 0.11 * eff_w + 0.034
        if nd < thr:
            nose_pen = max(0.06, nd / max(thr, 1e-6))

    return float(min(1.0, max(0.0, upper * split * nose_pen)))


def _pose_connection_pairs() -> list[tuple[int, int]]:
    try:
        import mediapipe as mp

        return [(int(a), int(b)) for a, b in mp.solutions.pose.POSE_CONNECTIONS]
    except Exception:
        return [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 7),
            (0, 4),
            (4, 5),
            (5, 6),
            (6, 8),
            (9, 10),
            (11, 12),
            (11, 13),
            (13, 15),
            (15, 17),
            (15, 19),
            (15, 21),
            (17, 19),
            (12, 14),
            (14, 16),
            (16, 18),
            (16, 20),
            (16, 22),
            (18, 20),
            (11, 23),
            (12, 24),
            (23, 24),
            (23, 25),
            (25, 27),
            (27, 29),
            (27, 31),
            (24, 26),
            (26, 28),
            (28, 30),
            (28, 32),
        ]


def _lm_vis(lm) -> float:
    v = getattr(lm, "visibility", None)
    if v is None:
        return 1.0
    return float(v)


def render_focus_chip(
    frame_bgr: np.ndarray,
    det: Detection,
    landmarks_full_norm: tuple[object, ...] | None,
    *,
    margin: float,
    line_vis_min: float = 0.35,
) -> np.ndarray | None:
    """Crop around YOLO bbox + margin; draw bbox + skeleton (full-frame normalized lms)."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(float, det.bbox)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pad = float(margin) * max(bw, bh)
    rx1 = int(max(0, np.floor(x1 - pad)))
    ry1 = int(max(0, np.floor(y1 - pad)))
    rx2 = int(min(w, np.ceil(x2 + pad)))
    ry2 = int(min(h, np.ceil(y2 + pad)))
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    chip = frame_bgr[ry1:ry2, rx1:rx2].copy()

    def to_chip(px: float, py: float) -> tuple[int, int]:
        return int(round(px - rx1)), int(round(py - ry1))

    X1, Y1 = to_chip(x1, y1)
    X2, Y2 = to_chip(x2, y2)
    cv2.rectangle(chip, (X1, Y1), (X2, Y2), (0, 220, 0), 2)

    if not landmarks_full_norm:
        return chip

    pts: list[tuple[int, int] | None] = []
    for lm in landmarks_full_norm:
        if lm.x is None or lm.y is None or _lm_vis(lm) < line_vis_min:
            pts.append(None)
            continue
        pxf = float(lm.x) * float(w)
        pyf = float(lm.y) * float(h)
        pts.append(to_chip(pxf, pyf))

    for a, b in _pose_connection_pairs():
        if a >= len(pts) or b >= len(pts):
            continue
        pa, pb = pts[a], pts[b]
        if pa is None or pb is None:
            continue
        cv2.line(chip, pa, pb, (0, 255, 255), 2, cv2.LINE_AA)

    return chip


def _normalize_gate_mode(raw: str) -> str:
    s = raw.strip().lower()
    return s if s in ("off", "yolo") else "yolo"


def _resolve_gate_mode(arg: str) -> str:
    if arg == "env":
        return _normalize_gate_mode(
            GATE_MODE if GATE_MODE in ("off", "yolo") else "yolo"
        )
    return _normalize_gate_mode(arg)


def make_peeing_detector() -> PeeingDetector:
    """Same kwargs as ``pipelines.test_pipeline``."""
    return PeeingDetector(
        pose_stride=PEEING_POSE_STRIDE,
        crop_margin=PEEING_CROP_MARGIN,
        min_visibility=PEEING_MIN_VISIBILITY,
        groin_dist_max=PEEING_GROIN_DIST_MAX,
        groin_loose_factor=PEEING_GROIN_LOOSE_FACTOR,
        wrist_band_min_visibility=PEEING_WRIST_BAND_MIN_VISIBILITY,
        pelvic_band_y_above=PEEING_PELVIC_BAND_Y_ABOVE,
        pelvic_band_y_below=PEEING_PELVIC_BAND_Y_BELOW,
        standing_y_margin=PEEING_STANDING_Y_MARGIN,
        window_sec=PEEING_WINDOW_SEC,
        pose_match_threshold=PEEING_POSE_MATCH_THRESHOLD,
        alarm_enter_hit_fraction=PEEING_ALARM_ENTER_HIT_FRACTION,
        alarm_exit_hit_fraction=PEEING_ALARM_EXIT_HIT_FRACTION,
        alarm_min_samples=PEEING_ALARM_MIN_SAMPLES,
        squat_hip_knee_gap_max=PEEING_SQUAT_HIP_KNEE_GAP_MAX,
        squat_depth_scale=PEEING_SQUAT_DEPTH_SCALE,
        model_path=PEEING_POSE_MODEL_PATH,
        model_url=PEEING_POSE_MODEL_URL,
    )


def _draw_instant_pee_pct(chip: np.ndarray, *, inst: float) -> None:
    """Single line: instant peeing heuristic as a whole-number percent (0–100%)."""
    ch = chip.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = float(max(0.42, min(0.78, ch / 380.0)))
    y = int(18 + ch * 0.02)
    pct = int(round(min(1.0, max(0.0, float(inst))) * 100.0))
    t = f"{pct}%"
    cv2.putText(chip, t, (6, y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(chip, t, (6, y), font, scale, (0, 255, 255), 2, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("outputs/peeing_snippets"),
        help="Directory for snippet images (default: outputs/peeing_snippets)",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=PEEING_CROP_MARGIN,
        help="Extra margin around YOLO bbox for the saved chip (default: PEEING_CROP_MARGIN "
        "from settings — same order of padding as the pose crop in production)",
    )
    parser.add_argument(
        "--format",
        choices=("png", "jpg"),
        default="png",
        help="Image format (default: png)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Instant score cutoff (**export-debug** only; default: pose_match_threshold)",
    )
    parser.add_argument(
        "--score-mode",
        choices=("production", "export-debug"),
        default="production",
        help="production: update + save every YOLO-gated frame (instant %% on chip). "
        "export-debug: legacy thresholded export-only score.",
    )
    parser.add_argument(
        "--gate-mode",
        choices=("env", "off", "yolo"),
        default="yolo",
        help="With --score-mode production: YOLO cadence (default yolo=stride gate; "
        "env=settings.GATE_MODE; off=every frame).",
    )
    parser.add_argument(
        "--no-experiment-cross-arm",
        action="store_true",
        help="Production export: show only MediaPipe instant score on chip (disable "
        "export-only opposite-wrist-to-opposite-hip blend).",
    )
    parser.add_argument(
        "--instant-hit-range",
        nargs=2,
        type=int,
        metavar=("FIRST", "LAST"),
        help="Only frames FIRST..LAST (inclusive). Production warms up from 0 then saves "
        "images in that window; export-debug seeks to FIRST. Requires ``-o`` output dir.",
    )
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.is_file():
        print("Video not found:", video, file=sys.stderr)
        return 2

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print("Could not open video:", video, file=sys.stderr)
        return 2

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frame_first = 0
    frame_last: int | None = (n_frames - 1) if n_frames > 0 else None
    if args.instant_hit_range is not None:
        lo, hi = int(args.instant_hit_range[0]), int(args.instant_hit_range[1])
        if lo < 0 or hi < lo:
            print("--instant-hit-range: need 0 <= FIRST <= LAST", file=sys.stderr)
            return 2
        if n_frames > 0 and hi >= n_frames:
            print(
                f"--instant-hit-range LAST={hi} past last frame index ({n_frames - 1}).",
                file=sys.stderr,
            )
            return 2
        frame_first, frame_last = lo, hi

    yolo = YoloDetector(conf_threshold=YOLO_CONFIDENCE)
    peeing = make_peeing_detector()
    thr = float(args.threshold) if args.threshold is not None else float(peeing.pose_match_threshold)

    n_saved = 0
    ext = ".jpg" if args.format == "jpg" else ".png"
    im_params = [int(cv2.IMWRITE_JPEG_QUALITY), 92] if args.format == "jpg" else []

    try:
        if args.score_mode == "export-debug":
            rng = args.instant_hit_range is not None
            if rng:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_first))
                span = frame_last - frame_first + 1
                pbar = tqdm(
                    total=span,
                    desc=f"export peeing (export-debug frames {frame_first}-{frame_last})",
                    unit="f",
                )
                frame_idx = frame_first
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
                pbar = tqdm(
                    total=n_frames or None,
                    desc="export peeing (export-debug)",
                    unit="f",
                )
                frame_idx = 0

            while True:
                if rng and frame_idx > frame_last:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                fd = FrameData(index=frame_idx, timestamp=frame_idx / fps, image=frame)
                dets = yolo.detect([fd])[0]
                scene = _filter_scene_detections(dets)

                h, w = frame.shape[:2]
                for slot, det in enumerate(peeing._person_detections(scene, YOLO_CONFIDENCE)):
                    box = peeing._clamp_crop(det.bbox, w, h)
                    if box is None:
                        continue
                    x1, y1, x2, y2 = box
                    crop = frame[y1:y2, x1:x2]
                    _, lms, _, _, _ = peeing._pose_on_crop(crop)
                    if not lms:
                        continue
                    dbg = _export_debug_instant_score(peeing, lms)
                    if dbg < thr:
                        continue
                    ovl = peeing._to_full_frame_landmarks(lms, x1, y1, x2, y2, w, h)
                    chip = render_focus_chip(
                        frame,
                        det,
                        ovl,
                        margin=args.crop_margin,
                    )
                    if chip is None:
                        continue
                    out_path = out_dir / f"peeing_instant_{frame_idx:08d}_p{slot}{ext}"
                    okw = (
                        cv2.imwrite(str(out_path), chip, im_params)
                        if im_params
                        else cv2.imwrite(str(out_path), chip)
                    )
                    if not okw:
                        print("imwrite failed:", out_path, file=sys.stderr)
                    else:
                        n_saved += 1

                frame_idx += 1
                pbar.update(1)
            pbar.close()
            extra = f" frames {frame_first}-{frame_last}" if rng else ""
            print(
                f"Saved {n_saved} image(s) under {out_dir.resolve()}{extra} "
                f"(score-mode=export-debug threshold={thr})"
            )
        else:
            from core.yolo_stride_gate import YoloStrideGate, YoloStrideGateConfig

            gate_mode = _resolve_gate_mode(args.gate_mode)
            gate: object | None = None
            if gate_mode == "yolo":
                gate = YoloStrideGate(
                    YoloStrideGateConfig(
                        coarse_stride=YOLO_COARSE_STRIDE,
                        dense_stride=YOLO_DENSE_STRIDE,
                        dense_idle_miss_streak=YOLO_DENSE_IDLE_MISS_STREAK,
                    )
                )

            rng = args.instant_hit_range is not None
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
            last_read = frame_last if rng else (10**12)
            pbar_total = (last_read + 1) if rng else (n_frames or None)
            pbar = tqdm(
                total=pbar_total,
                desc=(
                    f"export peeing (production gate={gate_mode} frames 0-{last_read})"
                    if rng
                    else f"export peeing (production gate={gate_mode})"
                ),
                unit="f",
            )
            frame_idx = 0
            while True:
                if rng and frame_idx > last_read:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                ts = frame_idx / fps
                run_yolo = True
                scene_dets: List[Detection] = []
                if gate_mode == "yolo":
                    assert gate is not None
                    run_yolo = gate.should_run_yolo(frame_idx)
                    if run_yolo:
                        fd = FrameData(index=frame_idx, timestamp=ts, image=frame)
                        raw = yolo.detect([fd])[0]
                        scene_dets = _filter_scene_detections(raw)
                        gate.observe(
                            frame_idx,
                            _scene_has_activity(scene_dets, YOLO_CONFIDENCE),
                        )
                else:
                    fd = FrameData(index=frame_idx, timestamp=ts, image=frame)
                    raw = yolo.detect([fd])[0]
                    scene_dets = _filter_scene_detections(raw)

                _ = peeing.update(
                    frame,
                    scene_dets,
                    run_yolo=run_yolo,
                    yolo_conf=YOLO_CONFIDENCE,
                    timestamp_sec=ts,
                )

                if frame_idx >= frame_first and run_yolo and scene_dets:
                    h, w = frame.shape[:2]
                    for slot, det in enumerate(
                        peeing._person_detections(scene_dets, YOLO_CONFIDENCE)
                    ):
                        box = peeing._clamp_crop(det.bbox, w, h)
                        if box is None:
                            continue
                        x1, y1, x2, y2 = box
                        crop = frame[y1:y2, x1:x2]
                        prod, lms, _, _, _ = peeing._pose_on_crop(crop)
                        if not lms:
                            continue
                        ovl = peeing._to_full_frame_landmarks(lms, x1, y1, x2, y2, w, h)
                        chip = render_focus_chip(
                            frame,
                            det,
                            ovl,
                            margin=args.crop_margin,
                        )
                        if chip is None:
                            continue
                        cross = _export_experimental_cross_wrist_opposite_hip(peeing, lms)
                        bend_g = _export_experimental_bend_groin_score(peeing, lms)
                        guard = _export_chip_pose_guard(peeing, lms)
                        if args.no_experiment_cross_arm:
                            display = min(1.0, float(prod) * guard)
                        else:
                            display = min(
                                1.0,
                                max(
                                    float(prod),
                                    _EXPERIMENT_CROSS_ARM_WEIGHT * cross,
                                    _EXPERIMENT_BEND_GROIN_WEIGHT * bend_g,
                                )
                                * guard,
                            )
                        _draw_instant_pee_pct(chip, inst=display)
                        out_path = out_dir / f"peeing_prod_{frame_idx:08d}_p{slot}{ext}"
                        okw = (
                            cv2.imwrite(str(out_path), chip, im_params)
                            if im_params
                            else cv2.imwrite(str(out_path), chip)
                        )
                        if not okw:
                            print("imwrite failed:", out_path, file=sys.stderr)
                        else:
                            n_saved += 1

                frame_idx += 1
                pbar.update(1)
            pbar.close()
            extra = (
                f" (warmup 0-{frame_first - 1}, saved window {frame_first}-{frame_last})"
                if rng and frame_first > 0
                else (f" (frames {frame_first}-{frame_last})" if rng else "")
            )
            print(
                f"Saved {n_saved} image(s) under {out_dir.resolve()}{extra} "
                f"(score-mode=production gate={gate_mode}; every YOLO frame in window)"
            )
    finally:
        cap.release()
        peeing.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
