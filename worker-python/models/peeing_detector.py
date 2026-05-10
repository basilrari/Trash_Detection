"""Peeing cue from MediaPipe Pose (Tasks API) on scene-YOLO person crops.

**Frame rule:** standing (hips above knees) and either wrist within ``hand_groin_y_threshold``
(normalized crop Y) of mid-groin — adapted from the standalone script heuristic.

**Temporal rule (stride-aware):** scene YOLO (and thus pose) runs only on stride-sampled frames.
Per calendar second of video time, each **tracked** person counts sampled frames and pose-hits.
A second counts as **positive** when hits ≥ ``min_hits_per_second`` (default 3 of ~5 samples).
After ``seconds_required`` consecutive positive seconds, that person is **confirmed** peeing.

No second YOLO call — uses detections already passed from the pipeline.

Requires the MediaPipe **Tasks** bundle (``pose_landmarker_*.task``).
"""

from __future__ import annotations

import logging
import math
import sys
import time
import urllib.request
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple


import cv2
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from models.types import Detection

logger = logging.getLogger(__name__)


class BlazePoseLandmark(IntEnum):
    """BlazePose 33-point indices for :class:`PoseLandmarker` (Tasks API)."""

    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_WRIST = 15
    RIGHT_WRIST = 16


MediaPipeMode = Literal["image", "video"]

PERSON_LABELS = ("person",)

OverlayLandmarks = Tuple[Tuple[NormalizedLandmark, ...], ...]
PeeingDisplayStatus = Literal["confirmed", "suspected"]


@dataclass(frozen=True)
class PeeingState:
    """Overlay / logging snapshot for one frame."""

    active: bool
    score: float
    sampled: bool
    frame_match: float
    status: PeeingDisplayStatus
    overlay_landmarks: OverlayLandmarks
    edge_enter: bool
    edge_exit: bool
    mark_bboxes: Tuple[Tuple[float, float, float, float], ...]
    mark_bboxes_suspected: Tuple[Tuple[float, float, float, float], ...]


def _ensure_pose_model_file(*, model_path: str, model_url: str) -> str:
    p = Path(model_path)
    if p.is_file() and p.stat().st_size > 0:
        return str(p.resolve())
    p.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading pose landmarker model to %s", p)
    tmp = p.with_suffix(p.suffix + ".part")
    try:
        urllib.request.urlretrieve(model_url, tmp)  # noqa: S310
        tmp.replace(p)
    except Exception:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise
    return str(p.resolve())


def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
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


@dataclass
class _PersonTrack:
    track_id: int
    bbox: Tuple[float, float, float, float]
    last_sample_ts: float
    bucket_sec: Optional[int] = None
    samples_in_bucket: int = 0
    hits_in_bucket: int = 0
    consecutive_good_seconds: int = 0
    latched_confirm: bool = False
    last_pose_timestamp_ms: int = -1


class PeeingDetector:
    def __init__(
        self,
        *,
        crop_margin: float = 0.12,
        min_visibility: float = 0.45,
        hand_groin_y_threshold: float = 0.1,
        min_hits_per_second: int = 3,
        seconds_required: int = 6,
        track_iou_threshold: float = 0.35,
        track_max_missed_seconds: float = 3.0,
        min_crop_side: int = 48,
        model_path: str | None = None,
        model_url: str | None = None,
        debug_timing: bool = False,
        mediapipe_mode: MediaPipeMode = "image",
        max_pose_persons_per_frame: int | None = None,
        min_person_box_height_px: float = 0.0,
    ) -> None:
        self.crop_margin = float(crop_margin)
        self.min_visibility = float(min_visibility)
        self.hand_groin_y_threshold = float(hand_groin_y_threshold)
        self.min_hits_per_second = max(1, int(min_hits_per_second))
        self.seconds_required = max(1, int(seconds_required))
        self.track_iou_threshold = float(track_iou_threshold)
        self.track_max_missed_seconds = float(max(0.5, track_max_missed_seconds))
        self.min_crop_side = int(min_crop_side)
        mode = str(mediapipe_mode).strip().lower()
        if mode not in ("image", "video"):
            raise ValueError(
                f"mediapipe_mode must be 'image' or 'video', got {mediapipe_mode!r}"
            )
        self._mediapipe_mode: MediaPipeMode = mode  # type: ignore[assignment]
        self.max_pose_persons_per_frame = (
            int(max_pose_persons_per_frame)
            if max_pose_persons_per_frame is not None
            else None
        )
        if self.max_pose_persons_per_frame is not None and self.max_pose_persons_per_frame < 1:
            raise ValueError("max_pose_persons_per_frame must be >= 1 when set")
        self.min_person_box_height_px = float(max(0.0, min_person_box_height_px))

        self._debug_timing = bool(debug_timing)
        self._timing_n = 0
        self._timing_sums: Dict[str, float] = {
            "to_rgb": 0.0,
            "wrap_image": 0.0,
            "detect": 0.0,
            "heuristic": 0.0,
        }
        self._dbg_run_yolo_updates = 0
        self._dbg_person_rows_total = 0
        self._dbg_pose_calls = 0
        self._dbg_crop_skips_small = 0
        self._dbg_hit_cap_skips = 0
        self._dbg_gate_skipped_short = 0
        self._dbg_gate_skipped_max_person = 0

        from mediapipe.tasks.python.core import base_options as mp_base_options
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )
        from mediapipe.tasks.python.vision.pose_landmarker import (
            PoseLandmarker,
            PoseLandmarkerOptions,
        )
        from mediapipe.tasks.python.vision.core import image as mp_image

        self._PoseLandmark = BlazePoseLandmark
        self._mp_image_mod = mp_image
        self._mp_base_options_mod = mp_base_options
        self._VisionTaskRunningMode = VisionTaskRunningMode
        self._PoseLandmarkerOptions = PoseLandmarkerOptions
        self._PoseLandmarker = PoseLandmarker

        cache_dir = Path.home() / ".cache" / "trash_detection_worker"
        default_path = cache_dir / "pose_landmarker_lite.task"
        default_url = (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        )
        self._resolved_model_path = _ensure_pose_model_file(
            model_path=model_path or str(default_path),
            model_url=model_url or default_url,
        )

        self._landmarker: Any = None
        self._video_lms: Dict[int, Any] = {}

        if self._mediapipe_mode == "image":
            self._landmarker = PoseLandmarker.create_from_model_path(
                self._resolved_model_path
            )

        self._tracks: List[_PersonTrack] = []
        self._next_id = 1
        self._had_any_confirmed = False

    def _make_video_landmarker(self) -> Any:
        opts = self._PoseLandmarkerOptions(
            base_options=self._mp_base_options_mod.BaseOptions(
                model_asset_path=self._resolved_model_path
            ),
            running_mode=self._VisionTaskRunningMode.VIDEO,
        )
        return self._PoseLandmarker.create_from_options(opts)

    def _dispose_track_landmarker(self, track_id: int) -> None:
        lm = self._video_lms.pop(track_id, None)
        if lm is not None:
            lm.close()

    def close(self) -> None:
        self._emit_debug_timing_report()
        # Finalize partial calendar-second buckets (EOF may land mid-second).
        for tr in list(self._tracks):
            if tr.bucket_sec is not None:
                self._flush_bucket(tr)
                tr.bucket_sec = None
                tr.samples_in_bucket = 0
                tr.hits_in_bucket = 0
        lm = getattr(self, "_landmarker", None)
        if lm is not None:
            lm.close()
            self._landmarker = None
        for tid in list(self._video_lms.keys()):
            self._dispose_track_landmarker(tid)

    def reset(self) -> None:
        for tid in list(self._video_lms.keys()):
            self._dispose_track_landmarker(tid)
        self._tracks.clear()
        self._next_id = 1
        self._had_any_confirmed = False
        self._timing_n = 0
        for k in self._timing_sums:
            self._timing_sums[k] = 0.0
        self._dbg_run_yolo_updates = 0
        self._dbg_person_rows_total = 0
        self._dbg_pose_calls = 0
        self._dbg_crop_skips_small = 0
        self._dbg_hit_cap_skips = 0
        self._dbg_gate_skipped_short = 0
        self._dbg_gate_skipped_max_person = 0

    def _emit_debug_timing_report(self) -> None:
        """Print timing/counters to stderr when enabled."""
        if not self._debug_timing:
            return
        print(
            "[peeing] debug counters: "
            f"run_yolo_updates={self._dbg_run_yolo_updates}  "
            f"person_rows_after_gates={self._dbg_person_rows_total}  "
            f"pose_calls={self._dbg_pose_calls}  "
            f"crop_too_small={self._dbg_crop_skips_small}  "
            f"hit_cap_skips={self._dbg_hit_cap_skips}  "
            f"gate_short_box={self._dbg_gate_skipped_short}  "
            f"gate_max_person_drop={self._dbg_gate_skipped_max_person}  "
            f"mediapipe_mode={self._mediapipe_mode}",
            file=sys.stderr,
        )
        if self._timing_n <= 0 and self._dbg_pose_calls <= 0:
            print(
                "[peeing] PEEING_DEBUG_TIMING: no pose inference ran — "
                "no qualifying person rows, all crops too small, or only hit-cap skips.",
                file=sys.stderr,
            )
            return
        if self._timing_n <= 0:
            return
        n = float(self._timing_n)
        ms = {k: 1000.0 * v / n for k, v in self._timing_sums.items()}
        total_ms = sum(ms.values())
        msg = (
            f"[peeing] MediaPipe pose timing (avg ms over {self._timing_n} crops): "
            f"bgr_to_rgb+contiguous={ms['to_rgb']:.2f}  "
            f"mp_Image()={ms['wrap_image']:.2f}  "
            f"detect()={ms['detect']:.2f}  "
            f"standing_groin_heuristic={ms['heuristic']:.2f}  "
            f"sum={total_ms:.2f}"
        )
        print(msg, file=sys.stderr)

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

    def _is_standing(self, lms: list, PL: type[BlazePoseLandmark]) -> bool:
        lh = lms[PL.LEFT_HIP.value]
        rh = lms[PL.RIGHT_HIP.value]
        lk = lms[PL.LEFT_KNEE.value]
        rk = lms[PL.RIGHT_KNEE.value]

        def ok(lm) -> bool:
            return self._lm_vis(lm) >= self.min_visibility

        if not (ok(lh) and ok(rh) and ok(lk) and ok(rk)):
            return False
        if lh.y is None or rh.y is None or lk.y is None or rk.y is None:
            return False
        return bool(float(lh.y) < float(lk.y) and float(rh.y) < float(rk.y))

    def _hand_near_groin(self, lms: list, PL: type[BlazePoseLandmark]) -> bool:
        lh = lms[PL.LEFT_HIP.value]
        rh = lms[PL.RIGHT_HIP.value]
        lw = lms[PL.LEFT_WRIST.value]
        rw = lms[PL.RIGHT_WRIST.value]

        def ok(lm) -> bool:
            return self._lm_vis(lm) >= self.min_visibility

        if not ok(lh) or not ok(rh) or lh.y is None or rh.y is None:
            return False
        groin_y = (float(lh.y) + float(rh.y)) * 0.5
        thr = self.hand_groin_y_threshold
        for w in (lw, rw):
            if not ok(w) or w.y is None:
                continue
            if abs(float(w.y) - groin_y) < thr:
                return True
        return False

    def _infer_pose_on_crop(
        self,
        crop_bgr: np.ndarray,
        landmarker: Any,
        tr: _PersonTrack,
        timestamp_sec: float,
    ) -> bool:
        """Run MediaPipe on one person crop; IMAGE uses ``detect``, VIDEO uses ``detect_for_video``."""
        if crop_bgr.size == 0 or crop_bgr.shape[0] < 16 or crop_bgr.shape[1] < 16:
            return False
        dbg = self._debug_timing
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        t1 = time.perf_counter()
        mp_image = self._mp_image_mod.Image(self._mp_image_mod.ImageFormat.SRGB, rgb)
        t2 = time.perf_counter()
        if self._mediapipe_mode == "video":
            ts_ms = int(timestamp_sec * 1000)
            if ts_ms <= tr.last_pose_timestamp_ms:
                ts_ms = tr.last_pose_timestamp_ms + 1
            tr.last_pose_timestamp_ms = ts_ms
            result = landmarker.detect_for_video(mp_image, ts_ms)
        else:
            result = landmarker.detect(mp_image)
        t3 = time.perf_counter()
        if not result.pose_landmarks:
            if dbg:
                self._timing_n += 1
                self._timing_sums["to_rgb"] += t1 - t0
                self._timing_sums["wrap_image"] += t2 - t1
                self._timing_sums["detect"] += t3 - t2
                self._timing_sums["heuristic"] += 0.0
            return False
        lms = list(result.pose_landmarks[0])
        PL = self._PoseLandmark
        ok = self._is_standing(lms, PL) and self._hand_near_groin(lms, PL)
        t4 = time.perf_counter()
        if dbg:
            self._timing_n += 1
            self._timing_sums["to_rgb"] += t1 - t0
            self._timing_sums["wrap_image"] += t2 - t1
            self._timing_sums["detect"] += t3 - t2
            self._timing_sums["heuristic"] += t4 - t3
        return ok

    def _expire_tracks(self, timestamp_sec: float) -> None:
        alive: List[_PersonTrack] = []
        for tr in self._tracks:
            if timestamp_sec - tr.last_sample_ts <= self.track_max_missed_seconds:
                alive.append(tr)
            else:
                self._dispose_track_landmarker(tr.track_id)
        self._tracks = alive

    def _score_progress(self) -> float:
        if not self._tracks:
            return 0.0
        return max(
            min(1.0, t.consecutive_good_seconds / float(self.seconds_required))
            for t in self._tracks
        )

    def _flush_bucket(self, tr: _PersonTrack) -> None:
        """Evaluate completed calendar second for hit streak."""
        if tr.bucket_sec is None:
            return
        good = tr.hits_in_bucket >= self.min_hits_per_second
        if good:
            tr.consecutive_good_seconds += 1
        else:
            tr.consecutive_good_seconds = 0
        if not tr.latched_confirm and tr.consecutive_good_seconds >= self.seconds_required:
            tr.latched_confirm = True

    def _advance_bucket(self, tr: _PersonTrack, sec: int) -> None:
        if tr.bucket_sec is None:
            tr.bucket_sec = sec
            tr.samples_in_bucket = 0
            tr.hits_in_bucket = 0
            return
        while sec > tr.bucket_sec:
            self._flush_bucket(tr)
            tr.bucket_sec += 1
            tr.samples_in_bucket = 0
            tr.hits_in_bucket = 0

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
        empty_lm: OverlayLandmarks = tuple()
        ts = float(timestamp_sec)
        sec = int(math.floor(ts + 1e-9))

        prev_active = self._had_any_confirmed
        self._expire_tracks(ts)

        if not run_yolo:
            active = any(t.latched_confirm for t in self._tracks)
            self._had_any_confirmed = active
            mark_c = tuple(t.bbox for t in self._tracks if t.latched_confirm)
            mark_s = tuple(
                t.bbox
                for t in self._tracks
                if not t.latched_confirm and t.consecutive_good_seconds > 0
            )
            status: PeeingDisplayStatus = "confirmed" if active else "suspected"
            prog = self._score_progress()
            if active:
                prog = 1.0
            return PeeingState(
                active=active,
                score=float(prog),
                sampled=False,
                frame_match=0.0,
                status=status,
                overlay_landmarks=empty_lm,
                edge_enter=active and not prev_active,
                edge_exit=prev_active and not active,
                mark_bboxes=mark_c,
                mark_bboxes_suspected=mark_s,
            )

        persons_raw = sorted(
            self._person_detections(detections, yolo_conf),
            key=lambda d: -d.confidence,
        )
        gated: List[Detection] = []
        min_h = self.min_person_box_height_px
        for d in persons_raw:
            if min_h > 1e-6:
                x1, y1, x2, y2 = d.bbox
                if float(y2) - float(y1) < min_h:
                    self._dbg_gate_skipped_short += 1
                    continue
            gated.append(d)

        if (
            self.max_pose_persons_per_frame is not None
            and len(gated) > self.max_pose_persons_per_frame
        ):
            drop = len(gated) - self.max_pose_persons_per_frame
            self._dbg_gate_skipped_max_person += drop
            persons = gated[: self.max_pose_persons_per_frame]
        else:
            persons = gated

        self._dbg_run_yolo_updates += 1
        self._dbg_person_rows_total += len(persons)

        used_track_ids: set[int] = set()
        frame_any_hit = False

        for d in persons:
            bbox = d.bbox
            best_tr: Optional[_PersonTrack] = None
            best_iou = 0.0
            for tr in self._tracks:
                if tr.track_id in used_track_ids:
                    continue
                iou = _iou_xyxy(tr.bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tr = tr
            if best_tr is not None and best_iou >= self.track_iou_threshold:
                tr = best_tr
                used_track_ids.add(tr.track_id)
                tr.bbox = bbox
                tr.last_sample_ts = ts
            else:
                tr = _PersonTrack(
                    track_id=self._next_id,
                    bbox=bbox,
                    last_sample_ts=ts,
                )
                self._next_id += 1
                self._tracks.append(tr)
                if self._mediapipe_mode == "video":
                    self._video_lms[tr.track_id] = self._make_video_landmarker()

            self._advance_bucket(tr, sec)
            tr.samples_in_bucket += 1

            if tr.hits_in_bucket >= self.min_hits_per_second:
                self._dbg_hit_cap_skips += 1
                continue

            box = self._clamp_crop(bbox, w, h)
            if box is None:
                self._dbg_crop_skips_small += 1
                continue

            if self._mediapipe_mode == "image":
                pose_lm = self._landmarker
            else:
                pose_lm = self._video_lms.get(tr.track_id)
                if pose_lm is None:
                    pose_lm = self._make_video_landmarker()
                    self._video_lms[tr.track_id] = pose_lm

            x1, y1, x2, y2 = box
            crop = frame_bgr[y1:y2, x1:x2]
            self._dbg_pose_calls += 1
            hit = self._infer_pose_on_crop(crop, pose_lm, tr, ts)
            if hit:
                tr.hits_in_bucket += 1
                frame_any_hit = True

        active = any(t.latched_confirm for t in self._tracks)
        self._had_any_confirmed = active

        mark_c = tuple(t.bbox for t in self._tracks if t.latched_confirm)
        mark_s = tuple(
            t.bbox
            for t in self._tracks
            if not t.latched_confirm and t.consecutive_good_seconds > 0
        )

        edge_enter = active and not prev_active
        edge_exit = prev_active and not active

        prog = self._score_progress()
        if active:
            prog = 1.0

        status = "confirmed" if active else "suspected"

        return PeeingState(
            active=active,
            score=float(prog),
            sampled=True,
            frame_match=1.0 if frame_any_hit else 0.0,
            status=status,
            overlay_landmarks=empty_lm,
            edge_enter=edge_enter,
            edge_exit=edge_exit,
            mark_bboxes=mark_c,
            mark_bboxes_suspected=mark_s,
        )
