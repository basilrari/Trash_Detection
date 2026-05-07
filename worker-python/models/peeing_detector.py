"""Heuristic peeing cue from MediaPipe Pose (Tasks API) on YOLO person crops.

On each pose sample, MediaPipe runs on **every** YOLO ``person`` crop; the strongest
instant score in that frame is recorded (no per-person IDs or tracking).

**Standing** — wrist proximity to hips / pelvic band. **Squat** — hip–knee depth (same
``PEEING`` label). **Straddle** penalty reduces motorbike / wide stance false positives.

**Alarm**: over the last ``window_sec`` seconds of pose samples, **more than**
``match_hit_fraction`` of those samples must have instant pose score ≥
``pose_match_threshold`` (defaults: 5s, **strictly** >60% of samples, score ≥ 0.6).

Requires the MediaPipe **Tasks** bundle (``pose_landmarker_*.task``). If the file at
``model_path`` is missing, it is downloaded once from ``model_url`` (see settings).
"""

from __future__ import annotations

import logging
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.types import Detection

logger = logging.getLogger(__name__)

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

PERSON_LABELS = ("person",)

OverlayLandmarks = Tuple[Tuple[NormalizedLandmark, ...], ...]


@dataclass(frozen=True)
class PeeingState:
    """Latest peeing heuristic output for overlay / logging.

    ``score`` is the fraction of pose samples in the current window that are hits
    (only meaningful once the window is full). ``frame_match`` is the latest instant
    pose score when ``sampled`` is True.
    """

    active: bool
    score: float
    sampled: bool
    frame_match: float
    overlay_landmarks: OverlayLandmarks
    edge_enter: bool
    edge_exit: bool
    mark_bboxes: Tuple[Tuple[float, float, float, float], ...]


def _ensure_pose_model_file(*, model_path: str, model_url: str) -> str:
    p = Path(model_path)
    if p.is_file() and p.stat().st_size > 0:
        return str(p.resolve())
    p.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading pose landmarker model to %s", p)
    tmp = p.with_suffix(p.suffix + ".part")
    try:
        urllib.request.urlretrieve(model_url, tmp)  # noqa: S310 — fixed vendor URL
        tmp.replace(p)
    except Exception:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise
    return str(p.resolve())


class PeeingDetector:
    def __init__(
        self,
        *,
        pose_stride: int = 2,
        crop_margin: float = 0.12,
        min_visibility: float = 0.45,
        groin_dist_max: float = 0.13,
        standing_y_margin: float = 0.03,
        window_sec: float = 5.0,
        pose_match_threshold: float = 0.6,
        match_hit_fraction: float = 0.6,
        min_crop_side: int = 48,
        model_path: str | None = None,
        model_url: str | None = None,
        groin_loose_factor: float = 1.22,
        wrist_band_min_visibility: float = 0.44,
        pelvic_band_y_above: float = -0.06,
        pelvic_band_y_below: float = 0.17,
        squat_hip_knee_gap_max: float = 0.09,
        squat_depth_scale: float = 0.11,
    ) -> None:
        self.pose_stride = max(1, int(pose_stride))
        self.crop_margin = float(crop_margin)
        self.min_visibility = float(min_visibility)
        self.groin_dist_max = float(groin_dist_max)
        self.groin_loose_factor = float(groin_loose_factor)
        self.wrist_band_min_visibility = float(wrist_band_min_visibility)
        self.pelvic_band_y_above = float(pelvic_band_y_above)
        self.pelvic_band_y_below = float(pelvic_band_y_below)
        self.standing_y_margin = float(standing_y_margin)
        self.window_sec = float(max(1e-6, window_sec))
        self.pose_match_threshold = float(pose_match_threshold)
        self.match_hit_fraction = float(
            max(0.0, min(1.0, match_hit_fraction))
        )
        self.min_crop_side = int(min_crop_side)
        self.squat_hip_knee_gap_max = float(squat_hip_knee_gap_max)
        self.squat_depth_scale = float(max(1e-6, squat_depth_scale))

        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmark
        from mediapipe.tasks.python.vision.core import image as mp_image

        self._PoseLandmarker = PoseLandmarker
        self._PoseLandmark = PoseLandmark
        self._mp_image_mod = mp_image

        cache_dir = Path.home() / ".cache" / "trash_detection_worker"
        default_path = cache_dir / "pose_landmarker_lite.task"
        default_url = (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        )
        resolved_path = _ensure_pose_model_file(
            model_path=model_path or str(default_path),
            model_url=model_url or default_url,
        )
        self._landmarker = PoseLandmarker.create_from_model_path(resolved_path)

        self._stride_counter = 0
        self._last_display_active = False
        self._mark_bboxes: Tuple[Tuple[float, float, float, float], ...] = tuple()
        self._pose_samples: Deque[tuple[float, bool]] = deque()

    def close(self) -> None:
        lm = getattr(self, "_landmarker", None)
        if lm is not None:
            lm.close()
            self._landmarker = None  # type: ignore[assignment]

    def reset(self) -> None:
        self._stride_counter = 0
        self._last_display_active = False
        self._mark_bboxes = tuple()
        self._pose_samples.clear()

    def _person_detections(
        self, detections: Sequence[Detection], yolo_conf: float
    ) -> List[Detection]:
        out: List[Detection] = []
        for d in detections:
            if d.label not in PERSON_LABELS or d.confidence < yolo_conf:
                continue
            out.append(d)
        return out

    def _clamp_crop(
        self, bbox: tuple[float, float, float, float], w: int, h: int
    ) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = map(float, bbox)
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        pad_x, pad_y = bw * self.crop_margin, bh * self.crop_margin
        xi1 = int(max(0, np.floor(x1 - pad_x)))
        yi1 = int(max(0, np.floor(y1 - pad_y)))
        xi2 = int(min(w, np.ceil(x2 + pad_x)))
        yi2 = int(min(h, np.ceil(y2 + pad_y)))
        if xi2 - xi1 < self.min_crop_side or yi2 - yi1 < self.min_crop_side:
            return None
        return xi1, yi1, xi2, yi2

    def _lm_vis(self, lm) -> float:
        v = getattr(lm, "visibility", None)
        if v is None:
            return 1.0
        return float(v)

    def _upright_standing(self, lms: list, PL) -> bool:
        lh = lms[PL.LEFT_HIP.value]
        rh = lms[PL.RIGHT_HIP.value]
        lk = lms[PL.LEFT_KNEE.value]
        rk = lms[PL.RIGHT_KNEE.value]

        def vis_ok(lm) -> bool:
            return self._lm_vis(lm) >= self.min_visibility

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
            hip_m = (float(lh.y) + float(rh.y)) * 0.5
            knee_m = (float(lk.y) + float(rk.y)) * 0.5
            return hip_m + 0.05 < knee_m

        leg_ok = False
        if vis_ok(lh) and vis_ok(lk) and lh.y is not None and lk.y is not None:
            if lh.y + self.standing_y_margin < lk.y:
                leg_ok = True
        if vis_ok(rh) and vis_ok(rk) and rh.y is not None and rk.y is not None:
            if rh.y + self.standing_y_margin < rk.y:
                leg_ok = True
        return leg_ok

    def _standing_pee_score(self, lms: list) -> float:
        PL = self._PoseLandmark

        def lm_at(idx: int):
            return lms[idx]

        if not self._upright_standing(lms, PL):
            return 0.0

        lh = lm_at(PL.LEFT_HIP.value)
        rh = lm_at(PL.RIGHT_HIP.value)
        lk = lm_at(PL.LEFT_KNEE.value)
        rk = lm_at(PL.RIGHT_KNEE.value)
        lw = lm_at(PL.LEFT_WRIST.value)
        rw = lm_at(PL.RIGHT_WRIST.value)
        ls = lm_at(PL.LEFT_SHOULDER.value)
        rs = lm_at(PL.RIGHT_SHOULDER.value)

        def vis_ok(lm) -> bool:
            return self._lm_vis(lm) >= self.min_visibility

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
        tight = max(self.groin_dist_max, 1e-6)
        loose = tight * self.groin_loose_factor

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
                if self._lm_vis(wrist) < self.wrist_band_min_visibility:
                    continue
                if wrist.x is None or wrist.y is None:
                    continue
                wx, wy = float(wrist.x), float(wrist.y)
                if wy < smy - 0.02:
                    continue
                if wy < gwy + self.pelvic_band_y_above:
                    continue
                if wy > gwy + self.pelvic_band_y_below:
                    continue
                if abs(wx - gwx) > body_w * 0.98:
                    continue
                band_score = max(band_score, 0.88)

        eff = max(best_prox, band_score * 0.88)
        if eff <= 0.0:
            return 0.0
        return float(min(1.0, 0.34 + 0.66 * eff))

    def _squat_score(self, lms: list) -> float:
        PL = self._PoseLandmark
        lh, rh = lms[PL.LEFT_HIP.value], lms[PL.RIGHT_HIP.value]
        lk, rk = lms[PL.LEFT_KNEE.value], lms[PL.RIGHT_KNEE.value]

        def vis_ok(lm) -> bool:
            return self._lm_vis(lm) >= self.min_visibility

        if not (
            vis_ok(lh)
            and vis_ok(rh)
            and vis_ok(lk)
            and vis_ok(rk)
            and lh.y is not None
            and rh.y is not None
            and lk.y is not None
            and rk.y is not None
        ):
            return 0.0

        hip_m = (float(lh.y) + float(rh.y)) * 0.5
        knee_m = (float(lk.y) + float(rk.y)) * 0.5
        if hip_m < knee_m - self.squat_hip_knee_gap_max:
            return 0.0
        depth = (hip_m - (knee_m - self.squat_hip_knee_gap_max)) / self.squat_depth_scale
        depth = float(np.clip(depth, 0.0, 1.0))
        return float(min(1.0, 0.38 + 0.58 * depth))

    def _straddle_penalty(self, lms: list) -> float:
        PL = self._PoseLandmark
        lh, rh = lms[PL.LEFT_HIP.value], lms[PL.RIGHT_HIP.value]
        lk, rk = lms[PL.LEFT_KNEE.value], lms[PL.RIGHT_KNEE.value]
        la, ra = lms[PL.LEFT_ANKLE.value], lms[PL.RIGHT_ANKLE.value]

        def vis_ok(lm, t: float) -> bool:
            return self._lm_vis(lm) >= t

        mult = 1.0
        if (
            vis_ok(lh, 0.35)
            and vis_ok(rh, 0.35)
            and lh.x is not None
            and rh.x is not None
            and vis_ok(lk, 0.35)
            and vis_ok(rk, 0.35)
            and lk.x is not None
            and rk.x is not None
        ):
            hip_w = abs(float(lh.x) - float(rh.x)) + 1e-6
            knee_span = abs(float(lk.x) - float(rk.x))
            if knee_span > max(0.36, hip_w * 2.05):
                mult *= 0.28

        if (
            vis_ok(la, 0.32)
            and vis_ok(ra, 0.32)
            and la.x is not None
            and ra.x is not None
            and vis_ok(lh, 0.32)
            and vis_ok(rh, 0.32)
            and lh.x is not None
            and rh.x is not None
        ):
            ankle_span = abs(float(la.x) - float(ra.x))
            hip_w = abs(float(lh.x) - float(rh.x)) + 1e-6
            if ankle_span > max(0.44, hip_w * 2.9):
                mult *= 0.22

        return mult

    def _pose_on_crop(
        self, crop_bgr: np.ndarray
    ) -> tuple[float, list | None, float, float, float]:
        """Returns ``(score, landmarks_or_none, standing, squat, straddle_mult)``."""
        if crop_bgr.size == 0 or crop_bgr.shape[0] < 16 or crop_bgr.shape[1] < 16:
            return 0.0, None, 0.0, 0.0, 1.0
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)

        mp_image = self._mp_image_mod.Image(self._mp_image_mod.ImageFormat.SRGB, rgb)
        result = self._landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return 0.0, None, 0.0, 0.0, 1.0

        lms = list(result.pose_landmarks[0])
        stand = self._standing_pee_score(lms)
        squat = self._squat_score(lms)
        smult = self._straddle_penalty(lms)
        score = max(stand, squat) * smult
        score = float(min(1.0, max(0.0, score)))
        return score, lms, stand, squat, smult

    def _to_full_frame_landmarks(
        self,
        lms: list,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[NormalizedLandmark, ...]:
        cw, ch = float(x2 - x1), float(y2 - y1)
        fw, fh = max(1, frame_w), max(1, frame_h)
        out: list[NormalizedLandmark] = []
        for lm in lms:
            if lm.x is None or lm.y is None:
                out.append(
                    NormalizedLandmark(
                        visibility=lm.visibility,
                        presence=lm.presence,
                        name=lm.name,
                    )
                )
                continue
            fx = (float(lm.x) * cw + float(x1)) / fw
            fy = (float(lm.y) * ch + float(y1)) / fh
            fx = min(1.0, max(0.0, fx))
            fy = min(1.0, max(0.0, fy))
            out.append(
                NormalizedLandmark(
                    x=fx,
                    y=fy,
                    z=lm.z,
                    visibility=lm.visibility,
                    presence=lm.presence,
                    name=lm.name,
                )
            )
        return tuple(out)

    def debug_analyze_crop(self, crop_bgr: np.ndarray) -> dict[str, Any]:
        """JSON-friendly pose + heuristic breakdown for one BGR crop (copy-paste / tuning)."""
        score, lms, stand, squat, smult = self._pose_on_crop(crop_bgr)
        out: dict[str, Any] = {
            "final_score": score,
            "standing_score": stand,
            "squat_score": squat,
            "straddle_multiplier": smult,
            "pose_match_threshold": self.pose_match_threshold,
            "counts_as_hit_now": bool(score >= self.pose_match_threshold),
            "landmarks_normalized_in_crop": None,
        }
        if lms is None:
            return out
        lrows: list[dict[str, Any]] = []
        for i, lm in enumerate(lms):
            row: dict[str, Any] = {"index": i}
            if lm.x is not None:
                row["x"] = round(float(lm.x), 6)
            if lm.y is not None:
                row["y"] = round(float(lm.y), 6)
            vz = getattr(lm, "z", None)
            if vz is not None:
                row["z"] = round(float(vz), 6)
            vv = getattr(lm, "visibility", None)
            if vv is not None:
                row["visibility"] = round(float(vv), 6)
            name = getattr(lm, "name", None)
            if name:
                row["name"] = str(name)
            lrows.append(row)
        out["landmarks_normalized_in_crop"] = lrows
        return out

    def debug_person_reports(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[Detection],
        *,
        yolo_conf: float,
    ) -> list[dict[str, Any]]:
        """One report dict per YOLO person (after ``yolo_conf`` filter), with pose landmarks."""
        h, w = frame_bgr.shape[:2]
        reports: list[dict[str, Any]] = []
        for j, d in enumerate(self._person_detections(detections, yolo_conf)):
            box = self._clamp_crop(d.bbox, w, h)
            rec: dict[str, Any] = {
                "person_index": j,
                "yolo_label": d.label,
                "yolo_confidence": float(d.confidence),
                "bbox_xyxy_global": [float(x) for x in d.bbox],
                "crop_used_xyxy": None,
            }
            if box is None:
                rec["error"] = "crop too small after padding"
                reports.append(rec)
                continue
            x1, y1, x2, y2 = box
            rec["crop_used_xyxy"] = [x1, y1, x2, y2]
            crop = frame_bgr[y1:y2, x1:x2]
            rec.update(self.debug_analyze_crop(crop))
            reports.append(rec)
        return reports

    def _trim_pose_samples(self, timestamp_sec: float) -> None:
        t_cut = float(timestamp_sec) - self.window_sec
        while self._pose_samples and self._pose_samples[0][0] < t_cut:
            self._pose_samples.popleft()

    def _window_active(self, timestamp_sec: float) -> tuple[bool, float]:
        self._trim_pose_samples(timestamp_sec)
        if not self._pose_samples:
            return False, 0.0
        oldest = self._pose_samples[0][0]
        if float(timestamp_sec) - oldest < self.window_sec - 1e-9:
            return False, 0.0
        total = len(self._pose_samples)
        hits = sum(1 for _, ok in self._pose_samples if ok)
        frac = hits / total
        active = frac > self.match_hit_fraction + 1e-12
        return active, float(frac)

    def _finalize(
        self,
        *,
        sampled: bool,
        frame_match: float,
        overlay: OverlayLandmarks,
        timestamp_sec: float,
    ) -> PeeingState:
        display_active, hit_frac = self._window_active(timestamp_sec)
        edge_enter = display_active and not self._last_display_active
        edge_exit = (not display_active) and self._last_display_active
        self._last_display_active = display_active

        if not display_active:
            self._mark_bboxes = tuple()

        overlay_out: OverlayLandmarks = overlay if display_active else tuple()

        return PeeingState(
            active=display_active,
            score=float(hit_frac),
            sampled=sampled,
            frame_match=float(frame_match),
            overlay_landmarks=overlay_out,
            edge_enter=edge_enter,
            edge_exit=edge_exit,
            mark_bboxes=tuple(self._mark_bboxes),
        )

    def update(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[Detection],
        *,
        run_yolo: bool,
        yolo_conf: float,
        timestamp_sec: float,
    ) -> PeeingState:
        h, w = frame_bgr.shape[:2]
        empty_overlay: OverlayLandmarks = tuple()

        if not run_yolo:
            return self._finalize(
                sampled=False, frame_match=0.0, overlay=empty_overlay, timestamp_sec=timestamp_sec
            )

        persons = self._person_detections(detections, yolo_conf)
        if not persons:
            return self._finalize(
                sampled=False, frame_match=0.0, overlay=empty_overlay, timestamp_sec=timestamp_sec
            )

        self._stride_counter += 1
        if (self._stride_counter % self.pose_stride) != 0:
            return self._finalize(
                sampled=False, frame_match=0.0, overlay=empty_overlay, timestamp_sec=timestamp_sec
            )

        entries: list[tuple[Detection, float, Optional[Tuple[NormalizedLandmark, ...]]]] = []
        for d in persons:
            box = self._clamp_crop(d.bbox, w, h)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            crop = frame_bgr[y1:y2, x1:x2]
            score, lms, _stand, _squat, _smult = self._pose_on_crop(crop)
            ov = self._to_full_frame_landmarks(lms, x1, y1, x2, y2, w, h) if lms else None
            entries.append((d, score, ov))

        ts = float(timestamp_sec)
        if not entries:
            self._pose_samples.append((ts, False))
            return self._finalize(
                sampled=True, frame_match=0.0, overlay=tuple(), timestamp_sec=timestamp_sec
            )

        scores = [e[1] for e in entries]
        best = max(scores) if scores else 0.0
        hit = best >= self.pose_match_threshold
        self._pose_samples.append((ts, hit))

        eps = 1e-4
        focus_min = 0.14
        if best >= focus_min:
            self._mark_bboxes = tuple(e[0].bbox for e in entries if e[1] + eps >= best)
            overlay_list = [e[2] for e in entries if e[2] is not None and e[1] + eps >= best]
        else:
            overlay_list = []
        overlay: OverlayLandmarks = tuple(overlay_list)

        return self._finalize(
            sampled=True, frame_match=best, overlay=overlay, timestamp_sec=timestamp_sec
        )
