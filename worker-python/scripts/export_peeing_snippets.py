#!/usr/bin/env python3
"""Save PNG/JPG crops when the peeing cue crosses a score threshold.

**Default (``--score-mode production``)** matches the worker: same ``PeeingDetector``
constructor kwargs as ``pipelines.test_pipeline``, ``PeeingDetector.update()`` on
every frame with ``--gate-mode`` (default ``yolo`` = YOLO stride gate; ``env`` =
``settings.GATE_MODE``), and the same
internal **pose_stride** sampling. On frames where ``update`` actually samples pose
(``PeeingState.sampled``), saves one chip per person whose **production** instant
score from ``_pose_on_crop`` is ``>= --threshold`` (default: ``pose_match_threshold``).

**Legacy (``--score-mode export-debug``)** — runs MediaPipe on **every** frame for
every person and uses the **export-only** instant score helpers below (stricter
upright, seated crush, wrist–hip overlap, …). Does **not** call ``update()``; useful
for tuning helpers before porting them to ``models/peeing_detector.py``.

Example::

  cd worker-python
  python scripts/export_peeing_snippets.py inputs/clip.mp4 -o outputs/peeing_snips
  python scripts/export_peeing_snippets.py inputs/clip.mp4 -o out --score-mode export-debug
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
from models.peeing_detector import PeeingDetector, PeeingState  # noqa: E402
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


def _draw_production_hud(chip: np.ndarray, *, pstate: PeeingState, inst: float) -> None:
    ch = chip.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = float(max(0.38, min(0.62, ch / 400.0)))
    y = int(16 + ch * 0.02)
    pct = int(round(float(pstate.score) * 100.0))
    t = f"{str(pstate.status).upper()} | window hits {pct}% | inst {inst:.2f} (auto)"
    cv2.putText(chip, t, (6, y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(chip, t, (6, y), font, scale, (0, 255, 255), 1, cv2.LINE_AA)


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
        help="Instant score cutoff (default: PeeingDetector pose_match_threshold / env)",
    )
    parser.add_argument(
        "--score-mode",
        choices=("production", "export-debug"),
        default="production",
        help="production: update() + gate + pose_stride, save on production instant score. "
        "export-debug: legacy every-frame export-only score (no update).",
    )
    parser.add_argument(
        "--gate-mode",
        choices=("env", "off", "yolo"),
        default="yolo",
        help="With --score-mode production: YOLO cadence (default yolo=stride gate; "
        "env=settings.GATE_MODE; off=every frame).",
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

    yolo = YoloDetector(conf_threshold=YOLO_CONFIDENCE)
    peeing = make_peeing_detector()
    thr = float(args.threshold) if args.threshold is not None else float(peeing.pose_match_threshold)

    n_saved = 0
    ext = ".jpg" if args.format == "jpg" else ".png"
    im_params = [int(cv2.IMWRITE_JPEG_QUALITY), 92] if args.format == "jpg" else []

    try:
        if args.score_mode == "export-debug":
            pbar = tqdm(total=n_frames or None, desc="export peeing (export-debug)", unit="f")
            frame_idx = 0
            while True:
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
            print(
                f"Saved {n_saved} image(s) under {out_dir} "
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

            pbar = tqdm(
                total=n_frames or None,
                desc=f"export peeing (production gate={gate_mode})",
                unit="f",
            )
            frame_idx = 0
            while True:
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

                pstate = peeing.update(
                    frame,
                    scene_dets,
                    run_yolo=run_yolo,
                    yolo_conf=YOLO_CONFIDENCE,
                    timestamp_sec=ts,
                )

                if pstate.sampled and scene_dets:
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
                        if not lms or prod < thr:
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
                        _draw_production_hud(chip, pstate=pstate, inst=float(prod))
                        out_path = (
                            out_dir
                            / f"peeing_prod_{frame_idx:08d}_p{slot}_i{prod:.2f}{ext}"
                        )
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
            print(
                f"Saved {n_saved} image(s) under {out_dir} "
                f"(score-mode=production gate={gate_mode} threshold={thr})"
            )
    finally:
        cap.release()
        peeing.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
