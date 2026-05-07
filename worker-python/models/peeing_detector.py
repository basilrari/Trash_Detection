"""Heuristic peeing cue from MediaPipe Pose (Tasks API) on YOLO person crops (prototype).

Runs only when the caller passes ``run_yolo=True`` (aligned with the YOLO stride gate)
and at least one ``person`` detection is present; pose is further subsampled by
``POSE_STRIDE`` on those eligible frames. Temporal smoothing uses an EMA on a
per-frame max score across people.

**Standing urination** uses wrist proximity to hips / pelvic band (tuned to reduce
walking / wide-stance / motorbike false positives).

**Squatting** (defecation posture) uses hip–knee depth; merged into the same score and
UI label ``PEEING``.

**Sustained alarm**: ``PeeingState.active`` is true only after ``min_active_duration_sec``
of consecutive EMA above ``active_threshold`` (hysteresis via ``ema_release_threshold``).

Requires the MediaPipe **Tasks** bundle (``pose_landmarker_*.task``). If the file at
``model_path`` is missing, it is downloaded once from ``model_url`` (see settings).
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.types import Detection

logger = logging.getLogger(__name__)

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

PERSON_LABELS = ("person",)

OverlayLandmarks = Tuple[Tuple[NormalizedLandmark, ...], ...]


@dataclass(frozen=True)
class PeeingState:
    """Latest peeing heuristic output for overlay / logging."""

    active: bool
    score: float
    sampled: bool
    frame_match: float
    overlay_landmarks: OverlayLandmarks
    edge_enter: bool
    edge_exit: bool
    # YOLO person xyxy (pixel) tied to the latest strong pose sample; used to tag chips.
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
        ema_alpha: float = 0.4,
        active_threshold: float = 0.62,
        ema_release_threshold: float = 0.44,
        min_active_duration_sec: float = 5.0,
        decay_no_yolo: float = 0.97,
        decay_no_person: float = 0.96,
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
        self.ema_alpha = float(ema_alpha)
        self.active_threshold = float(active_threshold)
        self.ema_release_threshold = float(ema_release_threshold)
        self.min_active_duration_sec = float(max(0.0, min_active_duration_sec))
        self.decay_no_yolo = float(decay_no_yolo)
        self.decay_no_person = float(decay_no_person)
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
        self._ema = 0.0
        self._last_display_active = False
        self._mark_bboxes: Tuple[Tuple[float, float, float, float], ...] = tuple()
        self._sustain_start_t: float | None = None

    def close(self) -> None:
        lm = getattr(self, "_landmarker", None)
        if lm is not None:
            lm.close()
            self._landmarker = None  # type: ignore[assignment]

    def reset(self) -> None:
        self._stride_counter = 0
        self._ema = 0.0
        self._last_display_active = False
        self._mark_bboxes = tuple()
        self._sustain_start_t = None

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
        """True when hips are clearly above knees (typical standing), not seated / straddle."""
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
        """Wrist near groin / pelvic band; only when clearly upright standing."""
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
        """Deep squat / hips dropped toward knees (defecation cue); same UI label as peeing."""
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
        # Hips near or below knee line in image = crouched / squat (y grows downward).
        if hip_m < knee_m - self.squat_hip_knee_gap_max:
            return 0.0
        depth = (hip_m - (knee_m - self.squat_hip_knee_gap_max)) / self.squat_depth_scale
        depth = float(np.clip(depth, 0.0, 1.0))
        return float(min(1.0, 0.38 + 0.58 * depth))

    def _straddle_penalty(self, lms: list) -> float:
        """Down-weight motorbike / wide straddle: large ankle or knee span vs hip width."""
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

    def _pose_on_crop(self, crop_bgr: np.ndarray) -> tuple[float, list | None]:
        """Run pose on one BGR crop; return (heuristic score, crop-normalized landmarks or None)."""
        if crop_bgr.size == 0 or crop_bgr.shape[0] < 16 or crop_bgr.shape[1] < 16:
            return 0.0, None
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)

        mp_image = self._mp_image_mod.Image(self._mp_image_mod.ImageFormat.SRGB, rgb)
        result = self._landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return 0.0, None

        lms = list(result.pose_landmarks[0])
        stand = self._standing_pee_score(lms)
        squat = self._squat_score(lms)
        score = max(stand, squat)
        score *= self._straddle_penalty(lms)
        score = float(min(1.0, max(0.0, score)))
        return score, lms

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

    def _update_sustained(self, timestamp_sec: float) -> bool:
        """Update sustain timer; return whether the public ``active`` flag should be on."""
        if self._ema >= self.active_threshold:
            if self._sustain_start_t is None:
                self._sustain_start_t = float(timestamp_sec)
        elif self._ema < self.ema_release_threshold:
            self._sustain_start_t = None

        if self._sustain_start_t is None or self._ema < self.active_threshold:
            return False
        if self.min_active_duration_sec <= 0.0:
            return True
        return (float(timestamp_sec) - self._sustain_start_t) >= self.min_active_duration_sec

    def _finalize(
        self,
        *,
        sampled: bool,
        frame_match: float,
        overlay: OverlayLandmarks,
        timestamp_sec: float,
    ) -> PeeingState:
        display_active = self._update_sustained(timestamp_sec)
        edge_enter = display_active and not self._last_display_active
        edge_exit = (not display_active) and self._last_display_active
        self._last_display_active = display_active

        if self._ema < self.ema_release_threshold:
            self._mark_bboxes = tuple()

        overlay_out: OverlayLandmarks = overlay if display_active else tuple()

        return PeeingState(
            active=display_active,
            score=float(self._ema),
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
            self._ema *= self.decay_no_yolo
            return self._finalize(
                sampled=False, frame_match=0.0, overlay=empty_overlay, timestamp_sec=timestamp_sec
            )

        persons = self._person_detections(detections, yolo_conf)
        if not persons:
            self._ema *= self.decay_no_person
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
            score, lms = self._pose_on_crop(crop)
            ov = self._to_full_frame_landmarks(lms, x1, y1, x2, y2, w, h) if lms else None
            entries.append((d, score, ov))

        if not entries:
            return self._finalize(
                sampled=True, frame_match=0.0, overlay=tuple(), timestamp_sec=timestamp_sec
            )

        scores = [e[1] for e in entries]
        best = max(scores) if scores else 0.0
        self._ema = self.ema_alpha * best + (1.0 - self.ema_alpha) * self._ema

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
