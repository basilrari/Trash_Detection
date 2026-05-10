"""Peeing cue from pose on scene-YOLO person crops (YOLO pose or MediaPipe Tasks).

**Frame rule:** standing (hips above knees) and either wrist within ``hand_groin_y_threshold``
(normalized crop Y) of mid-groin — same thresholds for both backends (COCO vs BlazePose indices).

**Temporal rule (stride-aware):** scene YOLO (and thus pose) runs only on stride-sampled frames.
Per calendar second of video time, each **tracked** person counts sampled frames and pose-hits.
A second counts as **positive** when hits ≥ ``min_hits_per_second`` (default 3 of ~5 samples).
After ``seconds_required`` consecutive positive seconds, that person is **confirmed** peeing.

Uses detections already passed from the pipeline (no second scene-YOLO call).

**YOLO pose:** runtime is a **batched** Ultralytics **TensorRT** ``.engine`` only (fixed batch, FP16 typical).
**MediaPipe:** Tasks bundle ``pose_landmarker_*.task``; GPU uses TFLite/EGL.
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

from models.types import Detection
from models.ultralytics_call_stats import UltralyticsCallStats, empty_ultralytics_call_stats

logger = logging.getLogger(__name__)


class BlazePoseLandmark(IntEnum):
    """BlazePose 33-point indices for MediaPipe :class:`PoseLandmarker` (Tasks API)."""

    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_WRIST = 15
    RIGHT_WRIST = 16


class Coco17Landmark(IntEnum):
    """COCO 17 keypoints (Ultralytics YOLO pose)."""

    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14


PoseBackend = Literal["mediapipe", "yolo"]
MediaPipeMode = Literal["image", "video"]

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


def _mediapipe_wheel_supports_delegate_flag() -> bool:
    """True if this ``mediapipe`` install passes ``BaseOptions.delegate`` into native code."""
    from mediapipe.tasks.python.core import base_options_c as boc

    fields = [f[0] for f in getattr(boc.BaseOptionsC, "_fields_", [])]
    return "delegate" in fields


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
        seconds_required: int = 10,
        track_iou_threshold: float = 0.35,
        track_max_missed_seconds: float = 3.0,
        min_crop_side: int = 48,
        pose_backend: PoseBackend = "yolo",
        # MediaPipe-only
        model_path: str | None = None,
        model_url: str | None = None,
        mediapipe_mode: MediaPipeMode = "image",
        mediapipe_delegate: str = "auto",
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
    ) -> None:
        self.crop_margin = float(crop_margin)
        self.min_visibility = float(min_visibility)
        self.hand_groin_y_threshold = float(hand_groin_y_threshold)
        self.min_hits_per_second = max(1, int(min_hits_per_second))
        self.seconds_required = max(1, int(seconds_required))
        self.track_iou_threshold = float(track_iou_threshold)
        self.track_max_missed_seconds = float(max(0.5, track_max_missed_seconds))
        self.min_crop_side = int(min_crop_side)

        backend = str(pose_backend).strip().lower()
        if backend not in ("mediapipe", "yolo"):
            raise ValueError(f"pose_backend must be 'mediapipe' or 'yolo', got {pose_backend!r}")
        self._pose_backend: PoseBackend = backend  # type: ignore[assignment]

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
        self._dbg_pose_batch_launches = 0
        self._dbg_pose_batch_in = 0
        self._dbg_pose_batch_padded = 0
        self._dbg_pose_batch_slack = 0
        self._dbg_pose_prefetch_windows = 0
        self._dbg_pose_prefetch_frames = 0
        self._dbg_pose_prefetch_crops = 0
        self._dbg_pose_prefetch_unused_hits = 0
        self._yolo_pose_trt_timing = bool(yolo_pose_trt_timing) and self._pose_backend == "yolo"
        self._yolo_pose_prefetch_debug = bool(yolo_pose_prefetch_debug) and self._pose_backend == "yolo"
        self._pose_runtime_tag = ""

        self._landmarker: Any = None
        self._video_lms: Dict[int, Any] = {}
        self._pose_delegate: Any | None = None
        self._mp_image_mod: Any = None
        self._mp_base_options_mod: Any = None
        self._VisionTaskRunningMode: Any = None
        self._PoseLandmarkerOptions: Any = None
        self._PoseLandmarker: Any = None

        self._yolo_pose: Any = None
        self._yolo_pose_imgsz = max(32, int(yolo_pose_imgsz))
        self._yolo_pose_batch_size = 1
        self._yolo_pose_trt_dynamic = False

        import torch

        if self._pose_backend == "yolo":
            from ultralytics import YOLO

            spec = yolo_pose_model
            if not spec or not str(spec).strip():
                raise ValueError("yolo_pose_model is required when pose_backend is 'yolo'")
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
                dev = "cuda:0" if torch.cuda.is_available() else "cpu"
            else:
                dev = str(yolo_pose_device).strip()
            self._yolo_pose_device = dev
            self._delegate_label = "cuda" if dev.startswith("cuda") else "cpu"
            if dev.startswith("cuda") and not torch.cuda.is_available():
                print(
                    "[peeing] YOLO pose device requests CUDA but torch.cuda.is_available() is False — "
                    "running on CPU.",
                    file=sys.stderr,
                )
                self._yolo_pose_device = "cpu"
                self._delegate_label = "cpu"
            self._pose_runtime_tag = (
                f"yolo:trt:b{self._yolo_pose_batch_size}:dyn{int(self._yolo_pose_trt_dynamic)}:"
                f"{self._yolo_pose_device}"
            )
            print(
                f"[peeing] YOLO pose weights={wp}  device={self._yolo_pose_device}  "
                f"batch={self._yolo_pose_batch_size}  trt_dynamic={self._yolo_pose_trt_dynamic}  (trt)",
                file=sys.stderr,
            )
        else:
            self._init_mediapipe(
                model_path=model_path,
                model_url=model_url,
                mediapipe_delegate=mediapipe_delegate,
            )

        self._tracks: List[_PersonTrack] = []
        self._next_id = 1
        self._had_any_confirmed = False

    def _init_mediapipe(
        self,
        *,
        model_path: str | None,
        model_url: str | None,
        mediapipe_delegate: str,
    ) -> None:
        from mediapipe.tasks.python.core import base_options as mp_base_options
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )
        from mediapipe.tasks.python.vision.pose_landmarker import (
            PoseLandmarker,
            PoseLandmarkerOptions,
        )
        from mediapipe.tasks.python.vision.core import image as mp_image

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
        resolved = _ensure_pose_model_file(
            model_path=model_path or str(default_path),
            model_url=model_url or default_url,
        )

        BO = mp_base_options.BaseOptions
        ctypes_gpu = _mediapipe_wheel_supports_delegate_flag()
        pref = str(mediapipe_delegate).strip().lower()
        if pref not in ("cpu", "gpu", "auto"):
            raise ValueError(
                "mediapipe_delegate must be 'cpu', 'gpu', or 'auto', "
                f"got {mediapipe_delegate!r}"
            )

        def base_opts(delegate_val: Any | None) -> Any:
            if delegate_val is None:
                return BO(model_asset_path=resolved)
            return BO(model_asset_path=resolved, delegate=delegate_val)

        def try_create_image(gpu: bool) -> Any:
            del_val = BO.Delegate.GPU if gpu else None
            opts = PoseLandmarkerOptions(
                base_options=base_opts(del_val),
                running_mode=VisionTaskRunningMode.IMAGE,
            )
            return PoseLandmarker.create_from_options(opts)

        def try_create_probe(gpu: bool) -> Any:
            return try_create_image(gpu)

        want_try_gpu = pref in ("gpu", "auto") and ctypes_gpu
        if pref in ("gpu", "auto") and not ctypes_gpu:
            print(
                "[peeing] This mediapipe wheel does not expose ``delegate`` in BaseOptions C API — "
                "GPU cannot be enabled. Install a newer mediapipe (e.g. pip install -U mediapipe). "
                "Docs: https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/BaseOptions",
                file=sys.stderr,
            )
            if pref == "gpu":
                raise RuntimeError(
                    "mediapipe_delegate=gpu requires a mediapipe build with GPU delegate support."
                )

        if self._mediapipe_mode == "image":
            if pref == "cpu" or not want_try_gpu:
                self._landmarker = try_create_image(False)
                self._pose_delegate = None
                self._delegate_label = "cpu"
            else:
                try:
                    self._landmarker = try_create_image(True)
                    self._pose_delegate = BO.Delegate.GPU
                    self._delegate_label = "gpu"
                except BaseException as exc:
                    print(
                        "[peeing] MediaPipe GPU PoseLandmarker failed; falling back to CPU. "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    self._landmarker = try_create_image(False)
                    self._pose_delegate = None
                    self._delegate_label = "cpu_fallback"
        else:
            if pref == "cpu" or not want_try_gpu:
                probe = try_create_probe(False)
                probe.close()
                self._pose_delegate = None
                self._delegate_label = "cpu"
            else:
                try:
                    probe = try_create_probe(True)
                    probe.close()
                    self._pose_delegate = BO.Delegate.GPU
                    self._delegate_label = "gpu"
                except BaseException as exc:
                    print(
                        "[peeing] MediaPipe GPU PoseLandmarker probe failed; using CPU. "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    probe = try_create_probe(False)
                    probe.close()
                    self._pose_delegate = None
                    self._delegate_label = "cpu_fallback"

        del BO

        self._resolved_model_path = resolved
        self._pose_runtime_tag = f"mediapipe:{self._mediapipe_mode}:{self._delegate_label}"

        if self._delegate_label == "gpu":
            print(
                "[peeing] MediaPipe Tasks delegate: GPU (TensorFlow Lite GPU delegate / EGL-GL path on Linux — "
                "not PyTorch CUDA). If you see SIGSEGV or EGL warnings, set mediapipe_delegate to cpu "
                "(native crashes are not catchable in Python).",
                file=sys.stderr,
            )

    def _pose_base_options(self) -> Any:
        BO = self._mp_base_options_mod.BaseOptions
        if self._pose_delegate is None:
            return BO(model_asset_path=self._resolved_model_path)
        return BO(model_asset_path=self._resolved_model_path, delegate=self._pose_delegate)

    def _make_video_landmarker(self) -> Any:
        opts = self._PoseLandmarkerOptions(
            base_options=self._pose_base_options(),
            running_mode=self._VisionTaskRunningMode.VIDEO,
        )
        return self._PoseLandmarker.create_from_options(opts)

    def _dispose_track_landmarker(self, track_id: int) -> None:
        lm = self._video_lms.pop(track_id, None)
        if lm is not None:
            lm.close()

    def close(self) -> None:
        self._emit_debug_timing_report()
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
        self._dbg_gate_skipped_max_person = 0
        self._dbg_pose_batch_launches = 0
        self._dbg_pose_batch_in = 0
        self._dbg_pose_batch_padded = 0
        self._dbg_pose_batch_slack = 0
        self._dbg_pose_prefetch_windows = 0
        self._dbg_pose_prefetch_frames = 0
        self._dbg_pose_prefetch_crops = 0
        self._dbg_pose_prefetch_unused_hits = 0

    def reset_inference_batch_stats(self) -> None:
        """Clear YOLO pose batch counters (call at pipeline start, like scene YOLO / LP)."""
        self._dbg_pose_batch_launches = 0
        self._dbg_pose_batch_in = 0
        self._dbg_pose_batch_padded = 0
        self._dbg_pose_batch_slack = 0

    def get_inference_batch_stats(self) -> UltralyticsCallStats:
        """Cumulative pose-crop batching stats (YOLO backend only)."""
        if self._pose_backend != "yolo":
            return empty_ultralytics_call_stats()
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
            f"pose_backend={self._pose_backend}  "
            f"pose_runtime={self._pose_runtime_tag}  "
            f"pose_batch_launches={self._dbg_pose_batch_launches}  "
            f"pose_batch_in={self._dbg_pose_batch_in}  "
            f"pose_batch_pad_slots={self._dbg_pose_batch_padded}  "
            f"pose_batch_max_slack={self._dbg_pose_batch_slack}  "
            f"pose_prefetch_windows={self._dbg_pose_prefetch_windows}  "
            f"pose_prefetch_frames={self._dbg_pose_prefetch_frames}  "
            f"pose_prefetch_crops={self._dbg_pose_prefetch_crops}  "
            f"pose_prefetch_unused_hits={self._dbg_pose_prefetch_unused_hits}",
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
        label = "YOLO pose" if self._pose_backend == "yolo" else "MediaPipe pose"
        msg = (
            f"[peeing] {label} timing (avg ms over {self._timing_n} crops): "
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

    def _is_standing_coco(self, xyn: np.ndarray, conf: np.ndarray) -> bool:
        L = Coco17Landmark
        idxs = (L.LEFT_HIP, L.RIGHT_HIP, L.LEFT_KNEE, L.RIGHT_KNEE)
        for i in idxs:
            if conf[int(i)] < self.min_visibility:
                return False
        return bool(
            float(xyn[int(L.LEFT_HIP), 1]) < float(xyn[int(L.LEFT_KNEE), 1])
            and float(xyn[int(L.RIGHT_HIP), 1]) < float(xyn[int(L.RIGHT_KNEE), 1])
        )

    def _hand_near_groin_coco(self, xyn: np.ndarray, conf: np.ndarray) -> bool:
        L = Coco17Landmark
        lh, rh = int(L.LEFT_HIP), int(L.RIGHT_HIP)
        if conf[lh] < self.min_visibility or conf[rh] < self.min_visibility:
            return False
        groin_y = (float(xyn[lh, 1]) + float(xyn[rh, 1])) * 0.5
        thr = self.hand_groin_y_threshold
        for wi in (int(L.LEFT_WRIST), int(L.RIGHT_WRIST)):
            if conf[wi] < self.min_visibility:
                continue
            if abs(float(xyn[wi, 1]) - groin_y) < thr:
                return True
        return False

    def _infer_pose_on_crop(
        self,
        crop_bgr: np.ndarray,
        tr: _PersonTrack,
        timestamp_sec: float,
    ) -> bool:
        if crop_bgr.size == 0 or crop_bgr.shape[0] < 16 or crop_bgr.shape[1] < 16:
            return False

        if self._pose_backend == "yolo":
            return self._infer_yolo_pose_batch([crop_bgr])[0]

        # --- MediaPipe ---
        landmarker: Any
        if self._mediapipe_mode == "image":
            landmarker = self._landmarker
        else:
            landmarker = self._video_lms.get(tr.track_id)
            if landmarker is None:
                landmarker = self._make_video_landmarker()
                self._video_lms[tr.track_id] = landmarker

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
        PL = BlazePoseLandmark
        ok = self._is_standing(lms, PL) and self._hand_near_groin(lms, PL)
        t4 = time.perf_counter()
        if dbg:
            self._timing_n += 1
            self._timing_sums["to_rgb"] += t1 - t0
            self._timing_sums["wrap_image"] += t2 - t1
            self._timing_sums["detect"] += t3 - t2
            self._timing_sums["heuristic"] += t4 - t3
        return ok

    def _yolo_pose_hit_from_result(self, r: Any) -> bool:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return False
        kp = r.keypoints
        if kp is None or len(kp) == 0:
            return False
        best_i = int(boxes.conf.argmax().item())
        if best_i >= len(kp):
            best_i = 0
        xyn = kp.xyn[best_i].cpu().numpy()
        conf = kp.conf[best_i].cpu().numpy()
        return self._is_standing_coco(xyn, conf) and self._hand_near_groin_coco(xyn, conf)

    def _infer_yolo_pose_batch(self, crops_bgr: List[np.ndarray]) -> List[bool]:
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

        hits: List[bool] = []
        for i in range(valid):
            t_h0 = time.perf_counter()
            ok = self._yolo_pose_hit_from_result(results[i])
            t_h1 = time.perf_counter()
            hits.append(ok)
            if dbg:
                self._timing_n += 1
                self._timing_sums["to_rgb"] += prep_per
                self._timing_sums["wrap_image"] += 0.0
                self._timing_sums["detect"] += det_per
                self._timing_sums["heuristic"] += t_h1 - t_h0
        return hits

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
        if self._pose_backend != "yolo" or not items:
            return {}
        self._dbg_pose_prefetch_windows += 1
        self._dbg_pose_prefetch_frames += len(items)
        out_map: Dict[int, List[Optional[bool]]] = {}
        all_crops: List[np.ndarray] = []
        placements: List[Tuple[int, int]] = []

        for frame_idx, frame_bgr, dets in items:
            persons = self._gated_pose_persons(dets, yolo_conf, record_gate_debug=False)
            h, w = frame_bgr.shape[:2]
            row: List[Optional[bool]] = [None] * len(persons)
            out_map[frame_idx] = row
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
            hits_chunk = self._infer_yolo_pose_batch(chunk)
            for k, hit in enumerate(hits_chunk):
                fi, jj = placements[offset + k]
                out_map[fi][jj] = hit
            offset += len(hits_chunk)
        return out_map

    def update(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[Detection],
        *,
        run_yolo: bool,
        yolo_conf: float,
        timestamp_sec: float,
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

        persons = self._gated_pose_persons(detections, yolo_conf, record_gate_debug=True)

        self._dbg_run_yolo_updates += 1
        self._dbg_person_rows_total += len(persons)

        use_prefetch = (
            self._pose_backend == "yolo"
            and precomputed_yolo_pose_hits is not None
            and len(precomputed_yolo_pose_hits) == len(persons)
        )

        used_track_ids: set[int] = set()
        frame_any_hit = False
        use_mp_video = self._pose_backend == "mediapipe" and self._mediapipe_mode == "video"
        yolo_pose_work: List[Tuple[_PersonTrack, np.ndarray]] = []

        for j, d in enumerate(persons):
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
                if use_mp_video:
                    self._video_lms[tr.track_id] = self._make_video_landmarker()

            self._advance_bucket(tr, sec)
            tr.samples_in_bucket += 1

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
            if self._pose_backend == "yolo":
                pc: Optional[bool] = (
                    precomputed_yolo_pose_hits[j]
                    if use_prefetch and precomputed_yolo_pose_hits is not None
                    else None
                )
                if pc is not None:
                    if pc:
                        tr.hits_in_bucket += 1
                        frame_any_hit = True
                else:
                    yolo_pose_work.append((tr, crop))
            else:
                hit = self._infer_pose_on_crop(crop, tr, ts)
                if hit:
                    tr.hits_in_bucket += 1
                    frame_any_hit = True

        if self._pose_backend == "yolo" and yolo_pose_work:
            B = self._yolo_pose_batch_size
            for start in range(0, len(yolo_pose_work), B):
                batch = yolo_pose_work[start : start + B]
                crops = [c for _, c in batch]
                hits = self._infer_yolo_pose_batch(crops)
                for (tr, _), hit in zip(batch, hits):
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
