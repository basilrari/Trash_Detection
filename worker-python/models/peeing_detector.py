"""Peeing cue from pose on scene-YOLO person crops (YOLO pose only).

**Frame rule:** standing on **either** side (that side's hip above knee, both visible) and wrist
within ``hand_groin_y_threshold`` (normalized crop Y) of mid-groin or same-side hip when one
hip is missing (COCO keypoints).

**Temporal rule (stride-aware):** scene YOLO (and thus pose) runs only on stride-sampled frames.
Per calendar second of video time, each **tracked** person counts sampled frames and pose-hits
only after a **bbox stillness** gate (no accumulation while the person is moving).
A second counts as **positive** when hits ≥ ``min_hits_per_second`` (default 3 of ~5 samples).
After ``seconds_required`` consecutive positive seconds, that person is **confirmed** peeing.

Uses detections already passed from the pipeline (no second scene-YOLO call).

**YOLO pose:** runtime is a **batched** Ultralytics **TensorRT** ``.engine`` only (fixed batch, FP16 typical).
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np

from models.types import Detection
from models.ultralytics_call_stats import UltralyticsCallStats
from models.peeing_motorcycle_gate import person_seated_on_motorcycle
from models.peeing_pose_viz import PosePersonViz
from models.peeing_stillness import bbox_is_still, iou_xyxy

class Coco17Landmark(IntEnum):
    """COCO 17 keypoints (Ultralytics YOLO pose)."""

    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14


PoseBackend = Literal["yolo"]

PERSON_LABELS = ("person",)

OverlayLandmarks = Tuple[Any, ...]
PeeingDisplayStatus = Literal["confirmed", "suspected"]


@dataclass(frozen=True)
class PeeingState:
    """Overlay / logging snapshot for one frame."""

    active: bool
    score: float
    sampled: bool
    frame_match: float
    status: PeeingDisplayStatus
    overlay_landmarks: Tuple[Tuple[Any, ...], ...]
    edge_enter: bool
    edge_exit: bool
    mark_bboxes: Tuple[Tuple[float, float, float, float], ...]
    mark_bboxes_suspected: Tuple[Tuple[float, float, float, float], ...]
    pose_viz: Tuple[PosePersonViz, ...] = field(default_factory=tuple)


def _worker_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_model_path(p: str) -> Path:
    """Resolve pose weights relative to ``worker-python/`` when not absolute / missing cwd hit."""
    pp = Path(p)
    if pp.is_file():
        return pp
    cand = _worker_root() / p
    if cand.is_file():
        return cand
    return pp


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
    still_prev_bbox: Optional[Tuple[float, float, float, float]] = None
    stationary_seconds: float = 0.0
    stationary_ready: bool = False


class PeeingDetector:
    def __init__(
        self,
        *,
        crop_margin: float = 0.12,
        min_visibility: float = 0.45,
        hand_groin_y_threshold: float = 0.1,
        min_hits_per_second: int = 3,
        seconds_required: int = 5,
        track_iou_threshold: float = 0.35,
        track_max_missed_seconds: float = 3.0,
        still_seconds_required: float = 1.0,
        still_max_center_motion: float = 0.035,
        still_max_size_change: float = 0.12,
        still_min_iou: float = 0.65,
        # YOLO pose-only (Ultralytics TensorRT ``.engine``; CUDA when torch sees a GPU unless overridden)
        yolo_pose_model: str | None = None,
        yolo_pose_imgsz: int = 640,
        yolo_pose_batch_size: int = 8,
        yolo_pose_trt_dynamic: bool = False,
        yolo_pose_device: str | None = None,
        yolo_pose_trt_timing: bool = False,
        yolo_pose_prefetch_debug: bool = False,
        debug_timing: bool = False,
        max_pose_persons_per_frame: int | None = None,
        motorcycle_exclusion_enabled: bool = True,
        motorcycle_labels: Tuple[str, ...] = ("motorcycle", "motorbike"),
        motorcycle_bbox_expand_x: float = 0.15,
        motorcycle_bbox_expand_y: float = 0.10,
        motorcycle_lower_body_fraction: float = 0.60,
        motorcycle_lower_overlap_threshold: float = 0.10,
        collect_pose_viz: bool = False,
        persist_pose_viz: bool | None = None,
    ) -> None:
        self.crop_margin = float(crop_margin)
        self.min_visibility = float(min_visibility)
        self.hand_groin_y_threshold = float(hand_groin_y_threshold)
        self.min_hits_per_second = max(1, int(min_hits_per_second))
        self.seconds_required = max(1, int(seconds_required))
        self.track_iou_threshold = float(track_iou_threshold)
        self.track_max_missed_seconds = float(max(0.5, track_max_missed_seconds))
        self.still_seconds_required = float(max(0.0, still_seconds_required))
        self.still_max_center_motion = float(still_max_center_motion)
        self.still_max_size_change = float(still_max_size_change)
        self.still_min_iou = float(still_min_iou)

        self.max_pose_persons_per_frame = (
            int(max_pose_persons_per_frame)
            if max_pose_persons_per_frame is not None
            else None
        )
        if self.max_pose_persons_per_frame is not None and self.max_pose_persons_per_frame < 1:
            raise ValueError("max_pose_persons_per_frame must be >= 1 when set")

        self._motorcycle_exclusion_enabled = bool(motorcycle_exclusion_enabled)
        self._motorcycle_labels = tuple(str(x) for x in motorcycle_labels)
        self._motorcycle_expand_x = float(motorcycle_bbox_expand_x)
        self._motorcycle_expand_y = float(motorcycle_bbox_expand_y)
        self._motorcycle_lower_body_fraction = float(motorcycle_lower_body_fraction)
        self._motorcycle_overlap_threshold = float(motorcycle_lower_overlap_threshold)

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
        self._dbg_gate_skipped_max_person = 0
        self._dbg_gate_skipped_motorcycle = 0
        self._dbg_pose_batch_launches = 0
        self._dbg_pose_batch_in = 0
        self._dbg_pose_batch_padded = 0
        self._dbg_pose_batch_slack = 0
        self._dbg_pose_prefetch_windows = 0
        self._dbg_pose_prefetch_frames = 0
        self._dbg_pose_prefetch_crops = 0
        self._dbg_pose_prefetch_unused_hits = 0
        self._dbg_still_not_ready_person_rows = 0
        self._dbg_motion_resets = 0
        self._dbg_pose_hits_ignored_not_still = 0
        self._prefetch_pose_details_by_frame = {}
        self._yolo_pose_trt_timing = bool(yolo_pose_trt_timing)
        self._yolo_pose_prefetch_debug = bool(yolo_pose_prefetch_debug)
        self._collect_pose_viz = bool(collect_pose_viz)
        self._persist_pose_viz = bool(
            persist_pose_viz
            if persist_pose_viz is not None
            else collect_pose_viz
        )
        self._last_pose_viz: Tuple[PosePersonViz, ...] = ()
        self._prefetch_pose_details_by_frame: Dict[int, List[Optional[Tuple[Any, ...]]]] = {}
        self._pose_runtime_tag = ""

        self._yolo_pose: Any = None
        self._yolo_pose_imgsz = max(32, int(yolo_pose_imgsz))
        self._yolo_pose_batch_size = 1
        self._yolo_pose_trt_dynamic = False

        import torch

        from ultralytics import YOLO

        spec = yolo_pose_model
        if not spec or not str(spec).strip():
            raise ValueError("yolo_pose_model is required")
        wp = _resolve_model_path(str(spec).strip())
        if wp.suffix.lower() != ".engine":
            raise ValueError(
                f"YOLO pose requires a TensorRT .engine file; got: {wp}\n"
                "Set PEEING_YOLO_POSE_MODEL to a fixed-batch pose engine path."
            )
        if not wp.is_file():
            raise FileNotFoundError(
                f"YOLO pose TensorRT engine not found: {wp}\n"
                "Place the pose .engine under worker-python/weights/ or pass an absolute path."
            )
        wp = wp.resolve()
        self._yolo_pose_batch_size = max(1, int(yolo_pose_batch_size))
        self._yolo_pose_trt_dynamic = bool(yolo_pose_trt_dynamic)

        self._yolo_pose = YOLO(str(wp))
        if yolo_pose_device is None:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "YOLO pose requires CUDA when PEEING_YOLO_POSE_DEVICE is None (defaults to cuda:0). "
                    "Use a CUDA-capable machine or set PEEING_YOLO_POSE_DEVICE explicitly."
                )
            dev = "cuda:0"
        else:
            dev = str(yolo_pose_device).strip()
        if dev.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"PEEING_YOLO_POSE_DEVICE is {dev!r} but torch.cuda.is_available() is False."
            )
        self._yolo_pose_device = dev
        self._delegate_label = "cuda" if dev.startswith("cuda") else "cpu"
        self._pose_runtime_tag = (
            f"yolo:trt:b{self._yolo_pose_batch_size}:dyn{int(self._yolo_pose_trt_dynamic)}:"
            f"{self._yolo_pose_device}"
        )
        print(
            f"[peeing] YOLO pose weights={wp}  device={self._yolo_pose_device}  "
            f"batch={self._yolo_pose_batch_size}  trt_dynamic={self._yolo_pose_trt_dynamic}  (trt)",
            file=sys.stderr,
        )

        self._tracks: List[_PersonTrack] = []
        self._next_id = 1
        self._had_any_confirmed = False

    def close(self) -> None:
        self._emit_debug_timing_report()
        for tr in list(self._tracks):
            if tr.bucket_sec is not None:
                self._flush_bucket(tr)
                tr.bucket_sec = None
                tr.samples_in_bucket = 0
                tr.hits_in_bucket = 0

    def reset(self) -> None:
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
        self._dbg_gate_skipped_max_person = 0
        self._dbg_gate_skipped_motorcycle = 0
        self._dbg_pose_batch_launches = 0
        self._dbg_pose_batch_in = 0
        self._dbg_pose_batch_padded = 0
        self._dbg_pose_batch_slack = 0
        self._dbg_pose_prefetch_windows = 0
        self._dbg_pose_prefetch_frames = 0
        self._dbg_pose_prefetch_crops = 0
        self._dbg_pose_prefetch_unused_hits = 0
        self._dbg_still_not_ready_person_rows = 0
        self._dbg_motion_resets = 0
        self._dbg_pose_hits_ignored_not_still = 0
        self._prefetch_pose_details_by_frame = {}
        self._last_pose_viz = ()

    def _pose_viz_for_output(
        self,
        fresh: Tuple[PosePersonViz, ...],
        *,
        had_person_rows: bool,
    ) -> Tuple[PosePersonViz, ...]:
        """Return viz for this frame; optionally hold last skeleton across stride skips."""
        if not self._collect_pose_viz:
            return ()
        if fresh:
            if self._persist_pose_viz:
                self._last_pose_viz = fresh
            return fresh
        if had_person_rows:
            self._last_pose_viz = ()
            return ()
        if self._persist_pose_viz:
            return self._last_pose_viz
        return ()

    def _bbox_still(
        self, prev: Tuple[float, float, float, float], bbox: Tuple[float, float, float, float]
    ) -> bool:
        return bbox_is_still(
            prev,
            bbox,
            min_iou=self.still_min_iou,
            max_center_motion_norm=self.still_max_center_motion,
            max_size_change=self.still_max_size_change,
        )

    def _update_stillness(
        self,
        tr: _PersonTrack,
        ref_prev: Optional[Tuple[float, float, float, float]],
        bbox: Tuple[float, float, float, float],
        dt_sample: float,
    ) -> None:
        """Update per-track bbox stillness; reset peeing streak on motion (unless latched)."""
        if tr.latched_confirm:
            tr.still_prev_bbox = bbox
            return

        if ref_prev is None:
            tr.still_prev_bbox = bbox
            tr.stationary_seconds = 0.0
            tr.stationary_ready = False
            return

        if not self._bbox_still(ref_prev, bbox):
            self._dbg_motion_resets += 1
            tr.stationary_seconds = 0.0
            tr.stationary_ready = False
            tr.consecutive_good_seconds = 0
            tr.hits_in_bucket = 0
            tr.still_prev_bbox = bbox
            return

        tr.stationary_seconds += max(0.0, float(dt_sample))
        tr.still_prev_bbox = bbox
        if tr.stationary_seconds >= self.still_seconds_required:
            tr.stationary_ready = True

    def reset_inference_batch_stats(self) -> None:
        """Clear YOLO pose batch counters (call at pipeline start, like scene YOLO / LP)."""
        self._dbg_pose_batch_launches = 0
        self._dbg_pose_batch_in = 0
        self._dbg_pose_batch_padded = 0
        self._dbg_pose_batch_slack = 0

    def get_inference_batch_stats(self) -> UltralyticsCallStats:
        """Cumulative pose-crop batching stats (YOLO backend only)."""
        return UltralyticsCallStats(
            self._dbg_pose_batch_in,
            self._dbg_pose_batch_launches,
            self._dbg_pose_batch_padded,
            self._dbg_pose_batch_slack,
        )

    def get_pose_cross_frame_prefetch_stats(self) -> Tuple[int, int, int, int]:
        """``(windows, sampled_frames, prefetched_crops, unused_hits)`` for cross-frame prefetch."""
        return (
            self._dbg_pose_prefetch_windows,
            self._dbg_pose_prefetch_frames,
            self._dbg_pose_prefetch_crops,
            self._dbg_pose_prefetch_unused_hits,
        )

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
            f"gate_max_person_drop={self._dbg_gate_skipped_max_person}  "
            f"gate_motorcycle_drop={self._dbg_gate_skipped_motorcycle}  "
            f"pose_runtime={self._pose_runtime_tag}  "
            f"pose_batch_launches={self._dbg_pose_batch_launches}  "
            f"pose_batch_in={self._dbg_pose_batch_in}  "
            f"pose_batch_pad_slots={self._dbg_pose_batch_padded}  "
            f"pose_batch_max_slack={self._dbg_pose_batch_slack}  "
            f"pose_prefetch_windows={self._dbg_pose_prefetch_windows}  "
            f"pose_prefetch_frames={self._dbg_pose_prefetch_frames}  "
            f"pose_prefetch_crops={self._dbg_pose_prefetch_crops}  "
            f"pose_prefetch_unused_hits={self._dbg_pose_prefetch_unused_hits}  "
            f"still_not_ready_person_rows={self._dbg_still_not_ready_person_rows}  "
            f"motion_resets={self._dbg_motion_resets}  "
            f"pose_hits_ignored_not_still={self._dbg_pose_hits_ignored_not_still}",
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
            f"[peeing] YOLO pose timing (avg ms over {self._timing_n} crops): "
            f"bgr_to_rgb+contiguous={ms['to_rgb']:.2f}  "
            f"mp_Image()={ms['wrap_image']:.2f}  "
            f"detect()={ms['detect']:.2f}  "
            f"standing_groin_heuristic={ms['heuristic']:.2f}  "
            f"sum={total_ms:.2f}"
        )
        print(msg, file=sys.stderr)

    def _motorcycle_bboxes(
        self, detections: Sequence[Detection], yolo_conf: float
    ) -> List[Tuple[float, float, float, float]]:
        if not self._motorcycle_exclusion_enabled:
            return []
        out: List[Tuple[float, float, float, float]] = []
        labels = self._motorcycle_labels
        for d in detections:
            if d.label not in labels or d.confidence < yolo_conf:
                continue
            out.append(d.bbox)
        return out

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
        if xi2 <= xi1 or yi2 <= yi1:
            return None
        return xi1, yi1, xi2, yi2

    def _side_standing_coco(
        self, xyn: np.ndarray, conf: np.ndarray, hip_i: int, knee_i: int
    ) -> bool:
        if conf[int(hip_i)] < self.min_visibility or conf[int(knee_i)] < self.min_visibility:
            return False
        return bool(float(xyn[int(hip_i), 1]) < float(xyn[int(knee_i), 1]))

    def _standing_sides_label(self, xyn: np.ndarray, conf: np.ndarray) -> str:
        L = Coco17Landmark
        sides: List[str] = []
        if self._side_standing_coco(xyn, conf, int(L.LEFT_HIP), int(L.LEFT_KNEE)):
            sides.append("L")
        if self._side_standing_coco(xyn, conf, int(L.RIGHT_HIP), int(L.RIGHT_KNEE)):
            sides.append("R")
        return ",".join(sides)

    def _is_standing_coco(self, xyn: np.ndarray, conf: np.ndarray) -> bool:
        L = Coco17Landmark
        return bool(
            self._side_standing_coco(xyn, conf, int(L.LEFT_HIP), int(L.LEFT_KNEE))
            or self._side_standing_coco(xyn, conf, int(L.RIGHT_HIP), int(L.RIGHT_KNEE))
        )

    def _hand_near_groin_coco(self, xyn: np.ndarray, conf: np.ndarray) -> bool:
        return self._min_wrist_groin_y_dist(xyn, conf) < self.hand_groin_y_threshold

    def _min_wrist_groin_y_dist(self, xyn: np.ndarray, conf: np.ndarray) -> float:
        L = Coco17Landmark
        lh, rh = int(L.LEFT_HIP), int(L.RIGHT_HIP)
        lw, rw = int(L.LEFT_WRIST), int(L.RIGHT_WRIST)
        min_vis = self.min_visibility
        left_hip_ok = conf[lh] >= min_vis
        right_hip_ok = conf[rh] >= min_vis
        best = 1e9

        if left_hip_ok and right_hip_ok:
            groin_y = (float(xyn[lh, 1]) + float(xyn[rh, 1])) * 0.5
            for wi in (lw, rw):
                if conf[wi] >= min_vis:
                    best = min(best, abs(float(xyn[wi, 1]) - groin_y))
        elif left_hip_ok and conf[lw] >= min_vis:
            best = min(best, abs(float(xyn[lw, 1]) - float(xyn[lh, 1])))
        elif right_hip_ok and conf[rw] >= min_vis:
            best = min(best, abs(float(xyn[rw, 1]) - float(xyn[rh, 1])))
        return best

    def _best_pose_keypoints_from_result(
        self, r: Any
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return None, None
        kp = r.keypoints
        if kp is None or len(kp) == 0:
            return None, None
        best_i = int(boxes.conf.argmax().item())
        if best_i >= len(kp):
            best_i = 0
        xyn = kp.xyn[best_i].cpu().numpy()
        conf = kp.conf[best_i].cpu().numpy()
        return xyn, conf

    def _yolo_pose_hit_from_result(self, r: Any) -> bool:
        xyn, conf = self._best_pose_keypoints_from_result(r)
        if xyn is None or conf is None:
            return False
        return self._is_standing_coco(xyn, conf) and self._hand_near_groin_coco(xyn, conf)

    def _yolo_pose_analyze_result(
        self, r: Any
    ) -> Tuple[bool, bool, float, Optional[np.ndarray], Optional[np.ndarray]]:
        xyn, conf = self._best_pose_keypoints_from_result(r)
        if xyn is None or conf is None:
            return False, False, 1e9, None, None
        standing = self._is_standing_coco(xyn, conf)
        dist = self._min_wrist_groin_y_dist(xyn, conf)
        hit = standing and dist < self.hand_groin_y_threshold
        return hit, standing, dist, xyn, conf

    def _keypoints_xy_frame(
        self,
        xyn: np.ndarray,
        conf: np.ndarray,
        box: Tuple[float, float, float, float],
        frame_w: int,
        frame_h: int,
    ) -> Tuple[Tuple[Tuple[float, float], ...], Tuple[float, ...]]:
        x1, y1, x2, y2 = box
        cw = max(1.0, float(x2 - x1))
        ch = max(1.0, float(y2 - y1))
        pts: List[Tuple[float, float]] = []
        for i in range(len(xyn)):
            pts.append((float(x1) + float(xyn[i, 0]) * cw, float(y1) + float(xyn[i, 1]) * ch))
        return tuple(pts), tuple(float(c) for c in conf)

    def _make_pose_person_viz(
        self,
        *,
        bbox: Tuple[float, float, float, float],
        tr: _PersonTrack,
        hit: bool,
        standing: bool,
        min_dist: float,
        xyn: Optional[np.ndarray],
        conf: Optional[np.ndarray],
        frame_w: int,
        frame_h: int,
    ) -> Optional[PosePersonViz]:
        if xyn is None or conf is None:
            return None
        kxy, kconf = self._keypoints_xy_frame(xyn, conf, bbox, frame_w, frame_h)
        return PosePersonViz(
            bbox=bbox,
            keypoints_xy=kxy,
            keypoints_conf=kconf,
            pose_hit=hit,
            standing=standing,
            min_wrist_groin_y=float(min_dist),
            still_ready=tr.stationary_ready,
            latched_confirm=tr.latched_confirm,
            consecutive_good_seconds=tr.consecutive_good_seconds,
            hits_in_bucket=tr.hits_in_bucket,
            standing_sides=self._standing_sides_label(xyn, conf),
        )

    def _infer_yolo_pose_batch_analyzed(
        self, crops_bgr: List[np.ndarray]
    ) -> List[Tuple[bool, bool, float, Optional[np.ndarray], Optional[np.ndarray]]]:
        if not crops_bgr:
            return []
        dbg = self._debug_timing
        crops_in: List[np.ndarray] = []
        prep = 0.0
        for c in crops_bgr:
            t_p0 = time.perf_counter()
            cc = c if c.flags["C_CONTIGUOUS"] else np.ascontiguousarray(c)
            t_p1 = time.perf_counter()
            prep += t_p1 - t_p0
            crops_in.append(cc)
        valid = len(crops_in)
        B = self._yolo_pose_batch_size
        padded = 0
        slack_here = 0
        chunk = crops_in
        short = max(0, B - valid)
        if not self._yolo_pose_trt_dynamic:
            if short > 0:
                pad_im = np.zeros_like(crops_in[0])
                chunk = crops_in + [pad_im] * short
                padded = short
        else:
            slack_here = short
        t1 = time.perf_counter()
        results = self._yolo_pose.predict(
            chunk,
            imgsz=self._yolo_pose_imgsz,
            verbose=False,
            device=self._yolo_pose_device,
        )
        if not isinstance(results, list):
            results = list(results)
        t2 = time.perf_counter()
        detect_dt = t2 - t1
        self._dbg_pose_batch_launches += 1
        self._dbg_pose_batch_in += valid
        self._dbg_pose_batch_padded += padded
        self._dbg_pose_batch_slack += slack_here

        if self._yolo_pose_trt_timing:
            print(
                f"[pose-TRT] launch={self._dbg_pose_batch_launches}  real_crops={valid}  "
                f"padded_dummy={padded}  max_batch_slack={slack_here}  tensors_in_chunk={len(chunk)}  "
                f"predict_ms={1000.0 * detect_dt:.3f}",
                file=sys.stderr,
            )
        det_per = detect_dt / valid if valid else 0.0
        prep_per = prep / valid if valid else 0.0

        analyzed: List[
            Tuple[bool, bool, float, Optional[np.ndarray], Optional[np.ndarray]]
        ] = []
        for i in range(valid):
            t_h0 = time.perf_counter()
            item = self._yolo_pose_analyze_result(results[i])
            t_h1 = time.perf_counter()
            analyzed.append(item)
            if dbg:
                self._timing_n += 1
                self._timing_sums["to_rgb"] += prep_per
                self._timing_sums["wrap_image"] += 0.0
                self._timing_sums["detect"] += det_per
                self._timing_sums["heuristic"] += t_h1 - t_h0
        return analyzed

    def _infer_yolo_pose_batch(self, crops_bgr: List[np.ndarray]) -> List[bool]:
        return [a[0] for a in self._infer_yolo_pose_batch_analyzed(crops_bgr)]

    def _expire_tracks(self, timestamp_sec: float) -> None:
        alive: List[_PersonTrack] = []
        for tr in self._tracks:
            if timestamp_sec - tr.last_sample_ts <= self.track_max_missed_seconds:
                alive.append(tr)
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

    def _gated_pose_persons(
        self,
        detections: Sequence[Detection],
        yolo_conf: float,
        *,
        record_gate_debug: bool,
    ) -> List[Detection]:
        persons_raw = sorted(
            self._person_detections(detections, yolo_conf),
            key=lambda d: -d.confidence,
        )
        moto_bbs = self._motorcycle_bboxes(detections, yolo_conf)
        if self._motorcycle_exclusion_enabled and moto_bbs:
            gated: List[Detection] = []
            for p in persons_raw:
                if person_seated_on_motorcycle(
                    p.bbox,
                    moto_bbs,
                    expand_x=self._motorcycle_expand_x,
                    expand_y=self._motorcycle_expand_y,
                    lower_body_fraction=self._motorcycle_lower_body_fraction,
                    overlap_threshold=self._motorcycle_overlap_threshold,
                ):
                    if record_gate_debug:
                        self._dbg_gate_skipped_motorcycle += 1
                    continue
                gated.append(p)
        else:
            gated = list(persons_raw)

        if (
            self.max_pose_persons_per_frame is not None
            and len(gated) > self.max_pose_persons_per_frame
        ):
            drop = len(gated) - self.max_pose_persons_per_frame
            if record_gate_debug:
                self._dbg_gate_skipped_max_person += drop
            return gated[: self.max_pose_persons_per_frame]
        return gated

    def prefetch_yolo_pose_hits_for_window(
        self,
        items: Sequence[Tuple[int, np.ndarray, Sequence[Detection]]],
        yolo_conf: float,
    ) -> Dict[int, List[Optional[bool]]]:
        """Infer YOLO pose on all gated person crops for the given sampled frames.

        Returns ``frame_idx -> list`` aligned to :meth:`_gated_pose_persons` ordering (same as
        :meth:`update`). Entries are ``None`` when the crop was invalid (too small after clamp).
        """
        if not items:
            return {}
        self._dbg_pose_prefetch_windows += 1
        self._dbg_pose_prefetch_frames += len(items)
        self._prefetch_pose_details_by_frame = {}
        out_map: Dict[int, List[Optional[bool]]] = {}
        all_crops: List[np.ndarray] = []
        placements: List[Tuple[int, int]] = []

        for frame_idx, frame_bgr, dets in items:
            persons = self._gated_pose_persons(dets, yolo_conf, record_gate_debug=False)
            h, w = frame_bgr.shape[:2]
            row: List[Optional[bool]] = [None] * len(persons)
            out_map[frame_idx] = row
            if self._collect_pose_viz:
                self._prefetch_pose_details_by_frame[frame_idx] = [None] * len(persons)
            for j, d in enumerate(persons):
                box = self._clamp_crop(d.bbox, w, h)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                crop = frame_bgr[y1:y2, x1:x2]
                all_crops.append(crop)
                placements.append((frame_idx, j))

        self._dbg_pose_prefetch_crops += len(all_crops)
        if self._yolo_pose_prefetch_debug:
            print(
                f"[pose-prefetch] windows_total={self._dbg_pose_prefetch_windows}  "
                f"sampled_frames={len(items)}  crops={len(all_crops)}",
                file=sys.stderr,
            )

        if not all_crops:
            return out_map

        B = self._yolo_pose_batch_size
        offset = 0
        while offset < len(all_crops):
            chunk = all_crops[offset : offset + B]
            analyzed_chunk = self._infer_yolo_pose_batch_analyzed(chunk)
            for k, item in enumerate(analyzed_chunk):
                fi, jj = placements[offset + k]
                hit = item[0]
                out_map[fi][jj] = hit
                if self._collect_pose_viz:
                    self._prefetch_pose_details_by_frame[fi][jj] = item
            offset += len(analyzed_chunk)
        return out_map

    def update(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[Detection],
        *,
        run_yolo: bool,
        yolo_conf: float,
        timestamp_sec: float,
        frame_index: Optional[int] = None,
        precomputed_yolo_pose_hits: Sequence[Optional[bool]] | None = None,
    ) -> PeeingState:
        h, w = frame_bgr.shape[:2]
        empty_lm: Tuple[Tuple[Any, ...], ...] = tuple()
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
                if not t.latched_confirm
                and t.stationary_ready
                and t.consecutive_good_seconds > 0
            )
            status: PeeingDisplayStatus = "confirmed" if active else "suspected"
            prog = self._score_progress()
            if active:
                prog = 1.0
            pose_out = (
                self._last_pose_viz
                if self._collect_pose_viz and self._persist_pose_viz
                else ()
            )
            return PeeingState(
                active=active,
                score=float(prog),
                sampled=False,
                frame_match=0.0,
                status=status,
                overlay_landmarks=empty_lm,
                pose_viz=pose_out,
                edge_enter=active and not prev_active,
                edge_exit=prev_active and not active,
                mark_bboxes=mark_c,
                mark_bboxes_suspected=mark_s,
            )

        persons = self._gated_pose_persons(detections, yolo_conf, record_gate_debug=True)

        self._dbg_run_yolo_updates += 1
        self._dbg_person_rows_total += len(persons)

        use_prefetch = (
            precomputed_yolo_pose_hits is not None
            and len(precomputed_yolo_pose_hits) == len(persons)
        )

        used_track_ids: set[int] = set()
        frame_any_hit = False
        yolo_pose_work: List[Tuple[int, _PersonTrack, np.ndarray, Tuple[float, float, float, float]]] = []
        pose_viz_list: List[PosePersonViz] = []
        prefetch_details: List[Optional[Tuple[Any, ...]]] | None = None
        if (
            self._collect_pose_viz
            and frame_index is not None
            and frame_index in self._prefetch_pose_details_by_frame
        ):
            prefetch_details = self._prefetch_pose_details_by_frame[frame_index]

        for j, d in enumerate(persons):
            bbox = d.bbox
            best_tr: Optional[_PersonTrack] = None
            best_iou = 0.0
            for tr in self._tracks:
                if tr.track_id in used_track_ids:
                    continue
                iou = iou_xyxy(tr.bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tr = tr
            if best_tr is not None and best_iou >= self.track_iou_threshold:
                tr = best_tr
                used_track_ids.add(tr.track_id)
                dt_sample = max(0.0, ts - tr.last_sample_ts)
                ref_prev = tr.still_prev_bbox if tr.still_prev_bbox is not None else tr.bbox
                tr.bbox = bbox
                tr.last_sample_ts = ts
                self._update_stillness(tr, ref_prev, bbox, dt_sample)
            else:
                tr = _PersonTrack(
                    track_id=self._next_id,
                    bbox=bbox,
                    last_sample_ts=ts,
                )
                self._next_id += 1
                self._tracks.append(tr)
                self._update_stillness(tr, None, bbox, 0.0)

            self._advance_bucket(tr, sec)
            tr.samples_in_bucket += 1

            if not tr.latched_confirm and not tr.stationary_ready:
                self._dbg_still_not_ready_person_rows += 1

            if tr.hits_in_bucket >= self.min_hits_per_second:
                if (
                    use_prefetch
                    and precomputed_yolo_pose_hits is not None
                    and precomputed_yolo_pose_hits[j] is not None
                ):
                    self._dbg_pose_prefetch_unused_hits += 1
                self._dbg_hit_cap_skips += 1
                continue

            box = self._clamp_crop(bbox, w, h)
            if box is None:
                self._dbg_crop_skips_small += 1
                continue

            x1, y1, x2, y2 = box
            crop = frame_bgr[y1:y2, x1:x2]
            self._dbg_pose_calls += 1
            pc: Optional[bool] = (
                precomputed_yolo_pose_hits[j]
                if use_prefetch and precomputed_yolo_pose_hits is not None
                else None
            )
            bbox_tuple = (float(x1), float(y1), float(x2), float(y2))
            pose_detail: Optional[Tuple[Any, ...]] = None
            if prefetch_details is not None and j < len(prefetch_details):
                pose_detail = prefetch_details[j]

            if pc is not None:
                if pc:
                    if tr.stationary_ready:
                        tr.hits_in_bucket += 1
                        frame_any_hit = True
                    else:
                        self._dbg_pose_hits_ignored_not_still += 1
                if self._collect_pose_viz and pose_detail is not None:
                    hit_p, standing_p, dist_p, xyn_p, conf_p = pose_detail
                    viz = self._make_pose_person_viz(
                        bbox=bbox_tuple,
                        tr=tr,
                        hit=bool(hit_p),
                        standing=bool(standing_p),
                        min_dist=float(dist_p),
                        xyn=xyn_p,
                        conf=conf_p,
                        frame_w=w,
                        frame_h=h,
                    )
                    if viz is not None:
                        pose_viz_list.append(viz)
            else:
                yolo_pose_work.append((j, tr, crop, bbox_tuple))

        if yolo_pose_work:
            B = self._yolo_pose_batch_size
            for start in range(0, len(yolo_pose_work), B):
                batch = yolo_pose_work[start : start + B]
                crops = [c for _, _, c, _ in batch]
                analyzed = self._infer_yolo_pose_batch_analyzed(crops)
                for (_, tr, _, bbox_tuple), item in zip(batch, analyzed):
                    hit, standing_p, dist_p, xyn_p, conf_p = item
                    if hit:
                        if tr.stationary_ready:
                            tr.hits_in_bucket += 1
                            frame_any_hit = True
                        else:
                            self._dbg_pose_hits_ignored_not_still += 1
                    if self._collect_pose_viz:
                        viz = self._make_pose_person_viz(
                            bbox=bbox_tuple,
                            tr=tr,
                            hit=hit,
                            standing=standing_p,
                            min_dist=dist_p,
                            xyn=xyn_p,
                            conf=conf_p,
                            frame_w=w,
                            frame_h=h,
                        )
                        if viz is not None:
                            pose_viz_list.append(viz)

        active = any(t.latched_confirm for t in self._tracks)
        self._had_any_confirmed = active

        mark_c = tuple(t.bbox for t in self._tracks if t.latched_confirm)
        mark_s = tuple(
            t.bbox
            for t in self._tracks
            if not t.latched_confirm
            and t.stationary_ready
            and t.consecutive_good_seconds > 0
        )

        edge_enter = active and not prev_active
        edge_exit = prev_active and not active

        prog = self._score_progress()
        if active:
            prog = 1.0

        status = "confirmed" if active else "suspected"
        pose_out = self._pose_viz_for_output(
            tuple(pose_viz_list),
            had_person_rows=bool(persons),
        )

        return PeeingState(
            active=active,
            score=float(prog),
            sampled=True,
            frame_match=1.0 if frame_any_hit else 0.0,
            status=status,
            overlay_landmarks=empty_lm,
            pose_viz=pose_out,
            edge_enter=edge_enter,
            edge_exit=edge_exit,
            mark_bboxes=mark_c,
            mark_bboxes_suspected=mark_s,
        )
