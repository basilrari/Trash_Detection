#!/usr/bin/env python3
"""
YOLO → RF-DETR trash → LP → OCR on a local video; writes an annotated MP4.
LP and OCR are batched; labels only (no bounding boxes). Scene YOLO is **person + road vehicles**.

Run from worker-python/:

  python worker.py
  python worker.py inputs/myvideo.mp4
  python -m pipelines.test_pipeline

**Scheduling:** automatic stride targets ``SCENE_YOLO_TARGET_FRAMES_PER_SECOND`` scene-YOLO frames
per **second of video** (FPS clamped for the formula), unless ``FRAME_SAMPLE_STRIDE_OVERRIDE`` is set.
Scene YOLO only on frames where ``frame_idx % stride == 0``, micro-batched with ``YOLO_MICRO_BATCH_SIZE``.
Skipped frames reuse carried scene boxes for peeing and cached plate redraw.
Production inputs are expected at **5–60 FPS** nominal (warning if the container reports otherwise).

**Output video** — ``OUTPUT_VIDEO_ENCODER`` in ``settings.py`` (``auto``/``nvenc`` require ffmpeg NVENC; ``mp4v`` is explicit CPU only).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from rich.console import Console

from models.base import TrashDetector
from models.yolo_detector import YoloDetector
from models.lp_detector import LpDetector
from models.ocr import Ocr, OcrRecognizeStats
from models.peeing_detector import PeeingDetector, PeeingState
from models.rfdetr_trt_trash import _trt_timing_enabled
from models.types import Detection, FrameData, LicensePlate
from pipelines.lp_batch_coordinator import LpBatchCoordinator, LpQueuedCrop
from settings import (
    ANNOTATOR_SMART_POSITION,
    CIGARETTE_ENGINE_PATH,
    FFMPEG_PATH,
    INPUT_VIDEO_FPS_MAX,
    INPUT_VIDEO_FPS_MIN,
    LP_BATCH_ENABLED,
    LP_BATCH_MAX_CROPS,
    LP_BATCH_MAX_LATENCY_FRAMES,
    LP_CONFIDENCE,
    LP_ENGINE_PATH,
    LP_LOCK_REFRESH_STRIDE,
    LP_TRT_BATCH_SIZE,
    LP_TRT_DYNAMIC,
    LP_TRT_IMAGE_SIZE,
    LP_VEHICLE_LP_STRIDE,
    NVENC_CQ,
    NVENC_PRESET,
    OCR_LOCK_CONFIDENCE,
    OUTPUT_VIDEO,
    OUTPUT_VIDEO_ENCODER,
    PEEING_CROP_MARGIN,
    PEEING_DEBUG_TIMING,
    PEEING_HAND_GROIN_Y_THRESHOLD,
    PEEING_MAX_POSE_PERSONS_PER_FRAME,
    PEEING_PERSIST_POSE_VIZ,
    PEEING_MIN_HITS_PER_SECOND,
    PEEING_MIN_VISIBILITY,
    PEEING_MOTORCYCLE_BBOX_EXPAND_X,
    PEEING_MOTORCYCLE_BBOX_EXPAND_Y,
    PEEING_MOTORCYCLE_EXCLUSION_ENABLED,
    PEEING_MOTORCYCLE_LABELS,
    PEEING_MOTORCYCLE_LOWER_BODY_FRACTION,
    PEEING_MOTORCYCLE_LOWER_OVERLAP_THRESHOLD,
    PEEING_SECONDS_REQUIRED,
    PEEING_STILL_MAX_CENTER_MOTION,
    PEEING_STILL_MAX_SIZE_CHANGE,
    PEEING_STILL_MIN_IOU,
    PEEING_STILL_SECONDS_REQUIRED,
    PEEING_TRACK_IOU_THRESHOLD,
    PEEING_TRACK_MAX_MISSED_SECONDS,
    PEEING_YOLO_POSE_BATCH_SIZE,
    PEEING_YOLO_POSE_CROSS_FRAME_BATCH,
    PEEING_YOLO_POSE_DEVICE,
    PEEING_YOLO_POSE_IMGSZ,
    PEEING_YOLO_POSE_MODEL,
    PEEING_YOLO_POSE_PREFETCH_DEBUG,
    PEEING_YOLO_POSE_TRT_DYNAMIC,
    PEEING_YOLO_POSE_TRT_TIMING,
    PIPELINE_READ_AHEAD_QUEUE_SIZE,
    PIPELINE_WRITE_QUEUE_SIZE,
    PLATE_CONFIDENCE,
    RF_DETR_CIGARETTE_EVERY_N_BATCHES,
    RF_DETR_MAX_QUEUE_LATENCY_FRAMES,
    RF_DETR_PREPROCESS_CUDA,
    RF_DETR_TRT_TIMING,
    TRASH_CONFIDENCE,
    TRASH_ENGINE_PATH,
    VIDEO_PATH,
    YOLO_CONFIDENCE,
    YOLO_ENGINE_PATH,
    YOLO_MICRO_BATCH_SIZE,
    YOLO_TRT_BATCH_SIZE,
    YOLO_TRT_DYNAMIC,
    YOLO_TRT_IMAGE_SIZE,
)

console = Console()

from pipelines.cuda_bootstrap import (
    _ensure_pytorch_cuda_kernels_work,
    _log_visible_torch_cuda_device,
    _log_model_ready,
)
from pipelines.frame_stride import _resolve_frame_sample_stride
from pipelines.peeing_shared import (
    PERSON_LABELS,
    VEHICLE_LABELS,
    _draw_peeing_overlay,
    _filter_scene_detections,
    _is_scene_detection,
)
from pipelines.video_io import (
    VideoWriterSink,
    _maybe_wrap_capture,
    _maybe_wrap_video_sink,
    _open_output_video_sink,
    _ReadAheadVideoCapture,
)


def _log_pipeline_run_configuration(
    *,
    video_path: str,
    width: int,
    height: int,
    fps: float,
    total_frames: int,
    output_video: str,
    sink_label: str,
    trash: TrashDetector,
    frame_stride: int,
    stride_detail: str,
) -> None:
    """TRT layout, thresholds, encoder, and RF-DETR flags (from ``settings``)."""
    te = Path(TRASH_ENGINE_PATH).resolve()
    ce = Path(CIGARETTE_ENGINE_PATH).resolve()
    rf_pre = "unknown"
    b0 = b1 = b2 = "?"
    heads = getattr(trash, "_heads", None)
    if heads:
        w0 = heads[0][0]
        rf_pre = (
            "PyTorch CUDA → TRT D2D input"
            if w0.uses_cuda_preprocess()
            else "NumPy + OpenCV CPU → TRT H2D"
        )
        b0, b1, b2 = (getattr(w0, "batch", "?"), getattr(w0, "height", "?"), getattr(w0, "width", "?"))
    trt_pre_disp = repr(RF_DETR_PREPROCESS_CUDA)
    trt_tim_disp = repr(RF_DETR_TRT_TIMING)
    ymb = max(1, int(YOLO_MICRO_BATCH_SIZE))
    win = max(frame_stride * ymb, ymb)
    console.print("[bold]Run configuration[/]")
    console.print(f"  Video  [dim]{video_path}[/]  →  {width}×{height} @ {fps:.3f} fps, {total_frames} frames")
    console.print(
        f"  Input    nominal FPS [{INPUT_VIDEO_FPS_MIN}, {INPUT_VIDEO_FPS_MAX}] "
        "(warning only if container FPS is outside)"
    )
    console.print(
        f"  Gate   [bold]effective stride={frame_stride}[/]  uniform scene-YOLO every {frame_stride} decoded "
        f"frame(s); read windows ≤{win} frames; ``YOLO_MICRO_BATCH_SIZE={ymb}`` batches sampled frames per "
        f"``detect()``; non-sampled frames reuse last scene boxes; RF-DETR only when scene activity "
        f"≥{YOLO_CONFIDENCE} on a sampled frame"
    )
    console.print(f"         [dim]{stride_detail}[/]")
    console.print(
        f"  Conf   YOLO={YOLO_CONFIDENCE}  trash_RF={TRASH_CONFIDENCE}  plate={PLATE_CONFIDENCE}"
    )
    console.print(f"  TRT    static batch={b0}  input {b1}×{b2}  preprocess: [cyan]{rf_pre}[/]")
    console.print(f"         trash head     [dim]{te}[/]")
    console.print(f"         cigarette head [dim]{ce}[/]")
    console.print(
        f"  Cfg    RF_DETR_PREPROCESS_CUDA={trt_pre_disp}  RF_DETR_TRT_TIMING={trt_tim_disp}  "
        f"PEEING_YOLO_POSE_TRT_TIMING={repr(PEEING_YOLO_POSE_TRT_TIMING)}  "
        f"PEEING_YOLO_POSE_CROSS_FRAME_BATCH={repr(PEEING_YOLO_POSE_CROSS_FRAME_BATCH)}  "
        f"PEEING_YOLO_POSE_PREFETCH_DEBUG={repr(PEEING_YOLO_POSE_PREFETCH_DEBUG)}"
    )
    console.print(
        f"  Encode OUTPUT_VIDEO_ENCODER={OUTPUT_VIDEO_ENCODER!r}  "
        f"NVENC_PRESET={NVENC_PRESET!r}  NVENC_CQ={NVENC_CQ}  FFMPEG_PATH={FFMPEG_PATH!r}"
    )
    console.print(f"  Output [dim]{Path(output_video).resolve()}[/]")
    console.print(f"  Writer [cyan]{sink_label}[/]")
    _pee_cfg = (
        f"{Path(PEEING_YOLO_POSE_MODEL).expanduser().resolve()}  "
        f"batch={PEEING_YOLO_POSE_BATCH_SIZE}  backend=yolo  "
        f"still_s>={PEEING_STILL_SECONDS_REQUIRED}  "
        f"still_iou>={PEEING_STILL_MIN_IOU}  "
        f"still_center<={PEEING_STILL_MAX_CENTER_MOTION}  "
        f"still_sizeDelta<={PEEING_STILL_MAX_SIZE_CHANGE}  "
        f"confirm_s={PEEING_SECONDS_REQUIRED}"
    )
    console.print(f"  Peeing [dim]{_pee_cfg}[/]")
    if _trt_timing_enabled() or bool(PEEING_YOLO_POSE_TRT_TIMING):
        parts = []
        if _trt_timing_enabled():
            parts.append("RF-DETR ``[TRT]``")
        if bool(PEEING_YOLO_POSE_TRT_TIMING):
            parts.append("YOLO pose ``[pose-TRT]``")
        console.print(
            f"  [yellow]Verbose TRT timing is on[/] — expect extra lines: {', '.join(parts)}."
        )
    console.print(
        f"  Pipeline YOLO_MICRO_BATCH_SIZE={YOLO_MICRO_BATCH_SIZE}  "
        f"LP_VEHICLE_LP_STRIDE={LP_VEHICLE_LP_STRIDE}  "
        f"PIPELINE_READ_AHEAD_QUEUE_SIZE={PIPELINE_READ_AHEAD_QUEUE_SIZE}  "
        f"PIPELINE_WRITE_QUEUE_SIZE={PIPELINE_WRITE_QUEUE_SIZE}"
    )
    console.print(
        f"  Scene YOLO TensorRT  [dim]{Path(YOLO_ENGINE_PATH).resolve()}[/]  "
        f"max_batch={YOLO_TRT_BATCH_SIZE}  dynamic={YOLO_TRT_DYNAMIC}  imgsz={YOLO_TRT_IMAGE_SIZE}"
    )
    console.print(
        f"  LP YOLO TensorRT     [dim]{Path(LP_ENGINE_PATH).resolve()}[/]  "
        f"max_batch={LP_TRT_BATCH_SIZE}  dynamic={LP_TRT_DYNAMIC}  imgsz={LP_TRT_IMAGE_SIZE}  conf≥{LP_CONFIDENCE}"
    )
    console.print(
        f"  OCR lock  text_conf≥{OCR_LOCK_CONFIDENCE}  requires internal '-'  "
        f"lp_location_refresh={LP_LOCK_REFRESH_STRIDE}"
    )
    console.print(
        f"  LP batch  enabled={LP_BATCH_ENABLED}  max_crops={LP_BATCH_MAX_CROPS}  "
        f"max_latency_frames={LP_BATCH_MAX_LATENCY_FRAMES}"
    )


@dataclass
class PipelineStepTimes:
    """Cumulative seconds spent in each stage (wall-clock, ``perf_counter``)."""

    init_sec: float = 0.0
    yolo_sec: float = 0.0
    trash_sec: float = 0.0
    # Source frames actually passed into ``detect_trash`` (after person/vehicle filter).
    rfdetr_input_frames: int = 0
    rfdetr_trt_batches: int = 0
    rfdetr_trt_padded_slots: int = 0
    # Scene YOLO + LP Ultralytics (``.pt`` or TRT): filled from detectors at end of run.
    yolo_input_frames: int = 0
    yolo_batch_launches: int = 0
    yolo_padded_slots: int = 0
    yolo_max_batch_slack: int = 0
    lp_input_crops: int = 0
    lp_batch_launches: int = 0
    lp_padded_slots: int = 0
    lp_max_batch_slack: int = 0
    peeing_sec: float = 0.0
    peeing_pose_input_crops: int = 0
    peeing_pose_batch_launches: int = 0
    peeing_pose_padded_slots: int = 0
    peeing_pose_max_batch_slack: int = 0
    peeing_pose_prefetch_windows: int = 0
    peeing_pose_prefetch_frames: int = 0
    peeing_pose_prefetch_crops: int = 0
    peeing_pose_prefetch_unused_hits: int = 0
    annotate_draw_sec: float = 0.0
    lp_sec: float = 0.0
    ocr_sec: float = 0.0
    video_write_sec: float = 0.0
    other_sec: float = 0.0  # open video, build writer, gate setup, chunk assembly
    # LP cross-frame coordinator (uniform stride + LP_BATCH_ENABLED); optional metrics.
    lp_coordinator_batches: int = 0
    lp_coordinator_latency_events: int = 0
    lp_coordinator_emit_barriers: int = 0
    lp_coordinator_queue_full_flushes: int = 0
    lp_coordinator_eof_rounds: int = 0
    ocr_recognize_calls: int = 0
    ocr_plates_submitted: int = 0
    ocr_locked_reuse_skips: int = 0
    ocr_prefilter_skipped_plates: int = 0

    def print_summary(self, *, wall_total_sec: float) -> None:
        ann = (
            self.annotate_draw_sec
            + self.lp_sec
            + self.ocr_sec
        )
        inf = (
            self.yolo_sec
            + self.trash_sec
            + self.peeing_sec
            + ann
            + self.video_write_sec
        )
        console.print("[bold]Step timings (cumulative)[/]")
        console.print(f"  Model init:           {self.init_sec:8.2f} s")
        console.print(f"  Other (I/O, chunks):  {self.other_sec:8.2f} s")
        console.print(f"  YOLO:                 {self.yolo_sec:8.2f} s")
        if self.yolo_input_frames > 0 and self.yolo_sec > 0:
            eff = self.yolo_input_frames / self.yolo_sec
            console.print(
                f"  [dim]Scene YOLO inputs: {self.yolo_input_frames:8d} frames "
                f"→ {eff:5.1f} eff. FPS (inputs ÷ YOLO time only)[/]"
            )
            console.print(
                "  [dim]Scene YOLO note:[/] ``batch_launches`` counts Ultralytics ``model()`` forwards "
                "(``.pt``: one per ``detect()``; ``.engine``: one per chunk). "
                "``padded (dummy rows)`` counts blank tensors only when ``YOLO_TRT_DYNAMIC=False``. "
                "With ``YOLO_TRT_DYNAMIC=True``, ``max_batch_slack`` is only ``Σ (max_batch - n_real)`` "
                "per launch (headroom vs ``YOLO_TRT_BATCH_SIZE``), not extra inferences."
            )
        if self.yolo_batch_launches > 0:
            avg_real = self.yolo_input_frames / max(self.yolo_batch_launches, 1)
            slack_part = (
                f"  max-batch slack: {self.yolo_max_batch_slack}"
                if self.yolo_max_batch_slack > 0
                else ""
            )
            console.print(
                f"  [dim]Scene YOLO batches:[/] {self.yolo_batch_launches}  "
                f"padded (dummy rows): {self.yolo_padded_slots}{slack_part}  "
                f"avg real frames/batch: {avg_real:.2f}"
            )
        console.print(f"  RF-DETR (trash):      {self.trash_sec:8.2f} s")
        if self.rfdetr_input_frames > 0 and self.trash_sec > 0:
            eff = self.rfdetr_input_frames / self.trash_sec
            console.print(
                f"  [dim]RF-DETR inputs:   {self.rfdetr_input_frames:8d} frames "
                f"→ {eff:5.1f} eff. FPS (inputs ÷ RF-DETR time only)[/]"
            )
            console.print(
                "  [dim]RF-DETR note:[/] ``[RF-DETR] … fps`` logs are per ``detect_trash`` call; "
                "eff. FPS above is the fair average. GPU preprocess is opt-in in ``settings.py`` "
                "(``RF_DETR_PREPROCESS_CUDA``); default CPU preprocess + two TRT heads dominate when "
                "``[TRT]`` preprocess ms is high."
            )
        if self.rfdetr_trt_batches > 0:
            avg_real = self.rfdetr_input_frames / max(self.rfdetr_trt_batches, 1)
            console.print(
                f"  [dim]RF-DETR TRT batches:[/] {self.rfdetr_trt_batches}  padded slots: "
                f"{self.rfdetr_trt_padded_slots}  avg real frames/batch: {avg_real:.2f}"
            )
        console.print(f"  Peeing (pose):        {self.peeing_sec:8.2f} s")
        if self.peeing_pose_input_crops > 0 and self.peeing_sec > 0:
            eff_p = self.peeing_pose_input_crops / self.peeing_sec
            console.print(
                f"  [dim]YOLO pose inputs:  {self.peeing_pose_input_crops:8d} person crops "
                f"→ {eff_p:5.1f} eff. FPS (crops ÷ Peeing time only)[/]"
            )
            console.print(
                "  [dim]YOLO pose note:[/] ``padded (dummy rows)`` only when "
                "``PEEING_YOLO_POSE_TRT_DYNAMIC=False`` (static TRT batch). "
                "``max_batch_slack`` sums per-launch headroom vs ``PEEING_YOLO_POSE_BATCH_SIZE`` when dynamic; "
                "``PEEING_YOLO_POSE_TRT_TIMING`` prints ``[pose-TRT]`` per Ultralytics forward."
            )
        if self.peeing_pose_batch_launches > 0:
            avg_pc = self.peeing_pose_input_crops / max(self.peeing_pose_batch_launches, 1)
            slack_pp = (
                f"  max-batch slack: {self.peeing_pose_max_batch_slack}"
                if self.peeing_pose_max_batch_slack > 0
                else ""
            )
            console.print(
                f"  [dim]YOLO pose batches:[/] {self.peeing_pose_batch_launches}  "
                f"padded (dummy rows): {self.peeing_pose_padded_slots}{slack_pp}  "
                f"avg real crops/batch: {avg_pc:.2f}"
            )
        if self.peeing_pose_prefetch_windows > 0:
            console.print(
                "  [dim]YOLO pose cross-frame:[/] "
                f"windows={self.peeing_pose_prefetch_windows}  "
                f"sampled_frames={self.peeing_pose_prefetch_frames}  "
                f"prefetched_crops={self.peeing_pose_prefetch_crops}  "
                f"unused_hits={self.peeing_pose_prefetch_unused_hits}"
            )
        console.print(f"  Annotate (draw only): {self.annotate_draw_sec:8.2f} s")
        console.print(f"  LP detect:            {self.lp_sec:8.2f} s")
        if self.lp_input_crops > 0 and self.lp_sec > 0:
            eff_lp = self.lp_input_crops / self.lp_sec
            console.print(
                f"  [dim]LP YOLO inputs:    {self.lp_input_crops:8d} crops "
                f"→ {eff_lp:5.1f} eff. FPS (crops ÷ LP time only)[/]"
            )
            console.print(
                "  [dim]LP YOLO note:[/] ``padded (dummy rows)`` only when ``LP_TRT_DYNAMIC=False``. "
                "``max_batch_slack`` (dynamic TRT) is informational headroom vs ``LP_TRT_BATCH_SIZE``, "
                "not duplicate crops."
            )
        if self.lp_batch_launches > 0:
            avg_lp = self.lp_input_crops / max(self.lp_batch_launches, 1)
            slack_lp = (
                f"  max-batch slack: {self.lp_max_batch_slack}"
                if self.lp_max_batch_slack > 0
                else ""
            )
            console.print(
                f"  [dim]LP YOLO batches:[/] {self.lp_batch_launches}  "
                f"padded (dummy rows): {self.lp_padded_slots}{slack_lp}  "
                f"avg real crops/batch: {avg_lp:.2f}"
            )
        if self.lp_coordinator_batches > 0 or self.lp_coordinator_emit_barriers > 0:
            console.print(
                f"  [dim]LP cross-frame:[/] coordinator_batches={self.lp_coordinator_batches}  "
                f"latency_flushes={self.lp_coordinator_latency_events}  "
                f"emit_barriers={self.lp_coordinator_emit_barriers}  "
                f"queue_full_flushes={self.lp_coordinator_queue_full_flushes}  "
                f"eof_rounds={self.lp_coordinator_eof_rounds}"
            )
        console.print(f"  OCR:                  {self.ocr_sec:8.2f} s")
        if (
            self.ocr_recognize_calls > 0
            or self.ocr_locked_reuse_skips > 0
            or self.ocr_prefilter_skipped_plates > 0
        ):
            avg_ms = (
                (self.ocr_sec / self.ocr_recognize_calls) * 1000.0
                if self.ocr_recognize_calls > 0
                else 0.0
            )
            console.print(
                "  [dim]OCR detail:[/] "
                f"recognize_calls={self.ocr_recognize_calls}  "
                f"plates_submitted={self.ocr_plates_submitted}  "
                f"locked_reuse_skips={self.ocr_locked_reuse_skips}  "
                f"prefilter_skipped={self.ocr_prefilter_skipped_plates}  "
                f"avg_ms/call={avg_ms:.1f}"
            )
        console.print(f"  Video write:          {self.video_write_sec:8.2f} s")
        console.print(f"  [dim]Sum (inference): {inf:8.2f} s[/]")
        console.print(f"  [bold]Wall clock total:   {wall_total_sec:8.2f} s[/]")


def _pipeline_inference_seconds(times: PipelineStepTimes) -> float:
    """Scene YOLO + RF-DETR + peeing + LP + OCR + annotate draw (excludes mux/write and setup I/O)."""

    return (
        times.yolo_sec
        + times.trash_sec
        + times.peeing_sec
        + times.lp_sec
        + times.ocr_sec
        + times.annotate_draw_sec
    )


def _pipeline_encode_io_seconds(times: PipelineStepTimes) -> float:
    return times.video_write_sec + times.other_sec


@dataclass
class PipelineModelBundle:
    """Models loaded once for batch runs (call ``cleanup()`` when finished)."""

    yolo: YoloDetector
    lp_detector: LpDetector
    ocr: Ocr
    trash: TrashDetector
    peeing: PeeingDetector
    init_sec: float

    def cleanup(self) -> None:
        try:
            self.peeing.close()
        except Exception:
            pass
        try:
            self.ocr.close()
        except Exception:
            pass


@dataclass
class VideoPipelineRecord:
    """One row of metrics for batch manifests."""

    input_path: str
    output_path: str
    success: bool
    error: str | None
    duration_sec: float
    fps: float
    width: int
    height: int
    total_frames: int
    wall_sec: float
    inference_sec: float
    encode_io_sec: float
    models_init_sec: float
    times: PipelineStepTimes | None


def load_pipeline_models() -> PipelineModelBundle:
    """Load scene YOLO, LP, OCR, RF-DETR trash, and PeeingDetector once."""

    _ensure_pytorch_cuda_kernels_work()
    _log_visible_torch_cuda_device()

    console.print("[bold]Models ready[/]")
    t0 = time.perf_counter()
    yolo = YoloDetector(conf_threshold=YOLO_CONFIDENCE)
    yolo_ready = (
        f"TensorRT [dim]{Path(YOLO_ENGINE_PATH).resolve()}[/]  "
        f"max_batch={YOLO_TRT_BATCH_SIZE} dynamic={YOLO_TRT_DYNAMIC}"
    )
    _log_model_ready("Scene YOLO", yolo_ready)
    lp_detector = LpDetector()
    lp_ready = (
        f"TensorRT [dim]{Path(LP_ENGINE_PATH).resolve()}[/]  "
        f"max_batch={LP_TRT_BATCH_SIZE} dynamic={LP_TRT_DYNAMIC}"
    )
    _log_model_ready("License-plate YOLO", lp_ready)
    ocr = Ocr()
    _log_model_ready("PaddleOCR", f"inference device={ocr.paddle_device}")
    trash = _load_trash_detector_required()
    heads = getattr(trash, "_heads", None)
    if heads:
        w0 = heads[0][0]
        _log_model_ready(
            "RF-DETR TensorRT",
            f"batch={w0.batch}  input {w0.height}×{w0.width}  "
            f"{'CUDA preprocess' if w0.uses_cuda_preprocess() else 'CPU preprocess'}",
        )
    else:
        _log_model_ready("RF-DETR", "loaded")
    try:
        peeing = PeeingDetector(
            crop_margin=PEEING_CROP_MARGIN,
            min_visibility=PEEING_MIN_VISIBILITY,
            hand_groin_y_threshold=PEEING_HAND_GROIN_Y_THRESHOLD,
            min_hits_per_second=PEEING_MIN_HITS_PER_SECOND,
            seconds_required=PEEING_SECONDS_REQUIRED,
            track_iou_threshold=PEEING_TRACK_IOU_THRESHOLD,
            track_max_missed_seconds=PEEING_TRACK_MAX_MISSED_SECONDS,
            still_seconds_required=PEEING_STILL_SECONDS_REQUIRED,
            still_max_center_motion=PEEING_STILL_MAX_CENTER_MOTION,
            still_max_size_change=PEEING_STILL_MAX_SIZE_CHANGE,
            still_min_iou=PEEING_STILL_MIN_IOU,
            yolo_pose_model=PEEING_YOLO_POSE_MODEL,
            yolo_pose_imgsz=PEEING_YOLO_POSE_IMGSZ,
            yolo_pose_batch_size=PEEING_YOLO_POSE_BATCH_SIZE,
            yolo_pose_trt_dynamic=PEEING_YOLO_POSE_TRT_DYNAMIC,
            yolo_pose_device=PEEING_YOLO_POSE_DEVICE,
            yolo_pose_trt_timing=PEEING_YOLO_POSE_TRT_TIMING,
            yolo_pose_prefetch_debug=PEEING_YOLO_POSE_PREFETCH_DEBUG,
            debug_timing=PEEING_DEBUG_TIMING,
            max_pose_persons_per_frame=PEEING_MAX_POSE_PERSONS_PER_FRAME,
            motorcycle_exclusion_enabled=PEEING_MOTORCYCLE_EXCLUSION_ENABLED,
            motorcycle_labels=PEEING_MOTORCYCLE_LABELS,
            motorcycle_bbox_expand_x=PEEING_MOTORCYCLE_BBOX_EXPAND_X,
            motorcycle_bbox_expand_y=PEEING_MOTORCYCLE_BBOX_EXPAND_Y,
            motorcycle_lower_body_fraction=PEEING_MOTORCYCLE_LOWER_BODY_FRACTION,
            motorcycle_lower_overlap_threshold=PEEING_MOTORCYCLE_LOWER_OVERLAP_THRESHOLD,
            persist_pose_viz=PEEING_PERSIST_POSE_VIZ,
        )
    except Exception as exc:
        console.print(
            "[red]PeeingDetector failed to initialize (required).[/]\n"
            "Ensure ``pip install ultralytics torch`` and a pose "
            "``PEEING_YOLO_POSE_MODEL`` TensorRT ``.engine`` path.\n"
            f"[dim]{exc}[/]"
        )
        raise SystemExit(2) from exc
    _pee_ready = f"YOLO pose [dim]{Path(PEEING_YOLO_POSE_MODEL).expanduser().resolve()}[/]"
    _log_model_ready(
        "PeeingDetector",
        _pee_ready,
    )
    _pee_hint_extra = (
        f"YOLO pose TensorRT/default: [dim]{Path(PEEING_YOLO_POSE_MODEL).name}[/]  "
        f"batch={PEEING_YOLO_POSE_BATCH_SIZE}  imgsz={PEEING_YOLO_POSE_IMGSZ}"
    )
    if PEEING_YOLO_POSE_DEVICE:
        _pee_hint_extra += f"; device={PEEING_YOLO_POSE_DEVICE!r}"
    console.print(
        "[dim]Peeing hint:[/] standing + hand near groin; "
        f"≥{PEEING_MIN_HITS_PER_SECOND} sampled pose hits per calendar second for "
        f"{PEEING_SECONDS_REQUIRED} consecutive seconds; IoU person tracks; "
        f"{_pee_hint_extra}."
    )
    t_init_done = time.perf_counter()
    init_sec = t_init_done - t0
    console.print(f"[dim]Model init wall time: {init_sec:.2f}s[/]")
    return PipelineModelBundle(
        yolo=yolo,
        lp_detector=lp_detector,
        ocr=ocr,
        trash=trash,
        peeing=peeing,
        init_sec=init_sec,
    )


def run_pipeline_video(
    bundle: PipelineModelBundle,
    video_path: str,
    output_video: str,
    *,
    per_video_times_init_sec: float,
    models_init_sec: float,
    abort_on_error: bool = True,
) -> VideoPipelineRecord:
    """Run the uniform-stride pipeline on one video using an existing model bundle.

    Does **not** call ``bundle.cleanup()`` (caller closes models after the last video).
    """

    def _fail(
        msg: str,
        *,
        wall_sec: float = 0.0,
        duration_sec: float = 0.0,
        fps: float = 0.0,
        width: int = 0,
        height: int = 0,
        total_frames: int = 0,
    ) -> VideoPipelineRecord:
        console.print(f"[red]{msg}[/]")
        if abort_on_error:
            m = msg.lower()
            if "not found" in m:
                sys.exit(2)
            if "invalid fps" in m:
                sys.exit(4)
            sys.exit(3)
        return VideoPipelineRecord(
            input_path=video_path,
            output_path=output_video,
            success=False,
            error=msg,
            duration_sec=duration_sec,
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            wall_sec=wall_sec,
            inference_sec=0.0,
            encode_io_sec=0.0,
            models_init_sec=models_init_sec,
            times=None,
        )

    wall_start_video = time.perf_counter()
    times = PipelineStepTimes()
    times.init_sec = per_video_times_init_sec

    if not os.path.exists(video_path):
        return _fail(f"Video not found: {video_path}", wall_sec=time.perf_counter() - wall_start_video)

    t_io = time.perf_counter()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _fail(f"Failed to open video: {video_path}", wall_sec=time.perf_counter() - wall_start_video)

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        try:
            cap.release()
        except Exception:
            pass
        return _fail(f"Invalid FPS: {fps}", wall_sec=time.perf_counter() - wall_start_video)

    if fps < float(INPUT_VIDEO_FPS_MIN) or fps > float(INPUT_VIDEO_FPS_MAX):
        console.print(
            f"[yellow]Warning:[/] reported FPS {fps:.2f} is outside the nominal input range "
            f"[{INPUT_VIDEO_FPS_MIN}, {INPUT_VIDEO_FPS_MAX}] — timing/stride math assumes {INPUT_VIDEO_FPS_MIN}–{INPUT_VIDEO_FPS_MAX} fps."
        )

    stride_n, stride_detail = _resolve_frame_sample_stride(fps)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = (total_frames / fps) if fps > 0 else 0.0
    console.print(
        f"[cyan]Video capture[/] opened  {width}×{height} @ {fps:.2f} fps  ({total_frames} frames)"
    )

    cap = _maybe_wrap_capture(cap, queue_size=PIPELINE_READ_AHEAD_QUEUE_SIZE)

    bundle.peeing.reset()

    yolo = bundle.yolo
    lp_detector = bundle.lp_detector
    ocr = bundle.ocr
    trash = bundle.trash
    peeing = bundle.peeing

    annots = _make_frame_annotators(width, height)
    lp_cache = VehicleLpOcrCache(
        LP_VEHICLE_LP_STRIDE,
        lp_lock_refresh_stride=LP_LOCK_REFRESH_STRIDE,
        ocr_lock_confidence=OCR_LOCK_CONFIDENCE,
    )
    lp_batch = LpBatchCoordinator(
        lp_detector=lp_detector,
        ocr=ocr,
        cache=lp_cache,
        times=times,
        max_crops=max(1, int(LP_BATCH_MAX_CROPS)),
        max_latency_frames=max(0, int(LP_BATCH_MAX_LATENCY_FRAMES)),
        enabled=bool(LP_BATCH_ENABLED),
    )

    out_dir = os.path.dirname(os.path.abspath(output_video))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out, sink_label = _open_output_video_sink(
        output_video,
        fps=fps,
        width=width,
        height=height,
    )
    out = _maybe_wrap_video_sink(out, queue_size=PIPELINE_WRITE_QUEUE_SIZE)
    _log_pipeline_run_configuration(
        video_path=video_path,
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        output_video=output_video,
        sink_label=sink_label,
        trash=trash,
        frame_stride=stride_n,
        stride_detail=stride_detail,
    )
    times.other_sec += time.perf_counter() - t_io

    yolo.reset_inference_batch_stats()
    lp_detector.reset_inference_batch_stats()
    peeing.reset_inference_batch_stats()

    try:
        _run_pipeline_uniform_stride_batched(
            cap=cap,
            out=out,
            fps=fps,
            total_frames=total_frames,
            yolo=yolo,
            lp_detector=lp_detector,
            ocr=ocr,
            trash=trash,
            times=times,
            annots=annots,
            peeing=peeing,
            lp_cache=lp_cache,
            stride=stride_n,
            lp_batch=lp_batch,
        )

        console.print(f"[green]Annotated video saved:[/] {output_video}")

    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            out.release()
        except Exception:
            pass

    wall_sec = time.perf_counter() - wall_start_video
    _ingest_ultralytics_pipeline_stats(times, yolo, lp_detector, peeing)
    inf_sec = _pipeline_inference_seconds(times)
    enc_sec = _pipeline_encode_io_seconds(times)

    return VideoPipelineRecord(
        input_path=video_path,
        output_path=output_video,
        success=True,
        error=None,
        duration_sec=duration_sec,
        fps=fps,
        width=width,
        height=height,
        total_frames=total_frames,
        wall_sec=wall_sec,
        inference_sec=inf_sec,
        encode_io_sec=enc_sec,
        models_init_sec=models_init_sec,
        times=times,
    )


def run_pipeline(video_path: str, output_video: str) -> None:
    """Process ``video_path`` and write annotated video to ``output_video``."""

    wall_start = time.perf_counter()

    if not os.path.exists(video_path):
        console.print(f"[red]Video not found:[/] {video_path}")
        sys.exit(2)

    bundle = load_pipeline_models()
    try:
        rec = run_pipeline_video(
            bundle,
            video_path,
            output_video,
            per_video_times_init_sec=bundle.init_sec,
            models_init_sec=bundle.init_sec,
            abort_on_error=True,
        )
    finally:
        bundle.cleanup()

    wall_total = time.perf_counter() - wall_start
    if rec.times is not None:
        rec.times.print_summary(wall_total_sec=wall_total)


def _ingest_ultralytics_pipeline_stats(
    times: PipelineStepTimes, yolo: YoloDetector, lp: LpDetector, peeing: PeeingDetector
) -> None:
    ys = yolo.get_inference_batch_stats()
    times.yolo_input_frames = ys.input_units
    times.yolo_batch_launches = ys.batch_launches
    times.yolo_padded_slots = ys.padded_slots
    times.yolo_max_batch_slack = ys.max_batch_slack
    ls = lp.get_inference_batch_stats()
    times.lp_input_crops = ls.input_units
    times.lp_batch_launches = ls.batch_launches
    times.lp_padded_slots = ls.padded_slots
    times.lp_max_batch_slack = ls.max_batch_slack
    ps = peeing.get_inference_batch_stats()
    times.peeing_pose_input_crops = ps.input_units
    times.peeing_pose_batch_launches = ps.batch_launches
    times.peeing_pose_padded_slots = ps.padded_slots
    times.peeing_pose_max_batch_slack = ps.max_batch_slack
    pw, pf, pc, pu = peeing.get_pose_cross_frame_prefetch_stats()
    times.peeing_pose_prefetch_windows = pw
    times.peeing_pose_prefetch_frames = pf
    times.peeing_pose_prefetch_crops = pc
    times.peeing_pose_prefetch_unused_hits = pu


def normalize_plate_text(raw: str) -> str:
    """Keep only letters and digits; spaces and other characters become ``-`` (collapsed)."""
    if not raw or not str(raw).strip():
        return ""
    s = re.sub(r"[^0-9A-Za-z]+", "-", str(raw).strip())
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def _plate_text_has_internal_dash(text: str) -> bool:
    """True when normalized OCR retained a separator between plate groups."""
    s = str(text).strip()
    return "-" in s[1:-1] if len(s) >= 3 else False


def _iou_xyxy_f(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ar = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    br = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ar + br - inter
    return inter / union if union > 0 else 0.0


class VehicleLpOcrCache:
    """Vehicle tracks with LP stride and high-confidence OCR text lock."""

    def __init__(
        self,
        stride: int,
        *,
        lp_lock_refresh_stride: int,
        ocr_lock_confidence: float,
    ) -> None:
        self.stride = max(1, int(stride))
        self.lp_lock_refresh_stride = max(1, int(lp_lock_refresh_stride))
        self.ocr_lock_confidence = float(ocr_lock_confidence)
        self._next_id = 0
        self._tracks: dict[int, dict[str, Any]] = {}

    def _new_track(self, fx: tuple[float, float, float, float]) -> dict[str, Any]:
        return {
            "box": fx,
            "plate_box": None,
            "last_lp": -10**9,
            "draw": [],
            "locked": False,
            "ocr_text": "",
            "ocr_conf": 0.0,
            "last_best_lp_conf": 0.0,
        }

    def _need_lp(self, tid: int, frame_idx: int) -> bool:
        tr = self._tracks[tid]
        dt = frame_idx - int(tr["last_lp"])
        interval = self.lp_lock_refresh_stride if tr.get("locked") else self.stride
        return dt >= interval

    def _same_plate_location(
        self,
        tr: dict[str, Any],
        plate_box: tuple[float, float, float, float],
    ) -> bool:
        prev = tr.get("plate_box")
        if prev is None:
            return False
        try:
            px1, py1, px2, py2 = map(float, prev)
            gx1, gy1, gx2, gy2 = map(float, plate_box)
        except Exception:
            return False
        iou = _iou_xyxy_f((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
        if iou >= 0.15:
            return True
        pcx, pcy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
        gcx, gcy = (gx1 + gx2) * 0.5, (gy1 + gy2) * 0.5
        scale = max(8.0, px2 - px1, py2 - py1, gx2 - gx1, gy2 - gy1)
        return abs(pcx - gcx) <= 0.75 * scale and abs(pcy - gcy) <= 0.75 * scale

    def _can_reuse_locked_ocr(
        self,
        tr: dict[str, Any],
        plate_box: tuple[float, float, float, float],
    ) -> bool:
        return bool(
            tr.get("locked")
            and str(tr.get("ocr_text", "")).strip()
            and float(tr.get("ocr_conf", 0.0)) >= self.ocr_lock_confidence
            and self._same_plate_location(tr, plate_box)
        )

    def _record_ocr_result(
        self,
        tr: dict[str, Any],
        *,
        lp_conf: float,
        plate_box: tuple[float, float, float, float],
        raw_text: str,
        ocr_conf: float,
    ) -> str:
        plate_text_norm = normalize_plate_text(str(raw_text))
        valid_text = bool(plate_text_norm) and ocr_conf >= PLATE_CONFIDENCE

        if valid_text and float(ocr_conf) >= float(tr.get("ocr_conf", 0.0)):
            tr["ocr_text"] = plate_text_norm
            tr["ocr_conf"] = float(ocr_conf)

        tr["plate_box"] = tuple(float(v) for v in plate_box)
        best_text = str(tr.get("ocr_text", "")).strip()
        best_conf = float(tr.get("ocr_conf", 0.0))
        if (
            best_text
            and best_conf >= self.ocr_lock_confidence
            and _plate_text_has_internal_dash(best_text)
        ):
            tr["locked"] = True

        if best_text:
            return f"{best_text} {best_conf:.2f}"
        return f"{ocr_conf:.2f}" if ocr_conf >= PLATE_CONFIDENCE else f"LP {lp_conf:.2f}"

    def apply_lp_chunk_results(
        self,
        chunk: list[LpQueuedCrop],
        plates_per_sub: Sequence[Sequence[LicensePlate]],
        *,
        lp_detector: LpDetector,
        ocr: Ocr,
        times: PipelineStepTimes,
    ) -> None:
        _ = lp_detector
        draw_by_tid: dict[int, list[tuple[float, float, float, float, str]]] = {}

        ocr_crops: list[np.ndarray] = []
        ocr_owner: list[tuple[int, int, float, tuple[float, float, float, float]]] = []
        # (tid, frame_idx, lp_conf, global_xyxy)

        for j, qc in enumerate(chunk):
            tid = qc.tid
            tr = self._tracks.get(tid)
            if tr is None:
                continue
            frame_idx = qc.frame_idx
            w, h = qc.frame_w, qc.frame_h
            vx1, vy1, vx2, vy2 = qc.vx1, qc.vy1, qc.vx2, qc.vy2
            vcrop = qc.vehicle_crop
            cw, ch = int(vcrop.shape[1]), int(vcrop.shape[0])
            if j >= len(plates_per_sub):
                tr["last_lp"] = frame_idx
                continue
            best_plate: LicensePlate | None = None
            for plate in plates_per_sub[j]:
                if plate.confidence < LP_CONFIDENCE:
                    continue
                if best_plate is None or plate.confidence > best_plate.confidence:
                    best_plate = plate
            if best_plate is None:
                tr["last_lp"] = frame_idx
                continue
            plate = best_plate
            pbox = clamp_bbox(plate.bbox, cw, ch)
            if pbox is None:
                tr["last_lp"] = frame_idx
                continue
            px1, py1, px2, py2 = pbox
            plate_crop = vcrop[py1:py2, px1:px2]
            if plate_crop.size == 0:
                tr["last_lp"] = frame_idx
                continue
            gx1, gy1, gx2, gy2 = vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2
            gbox = clamp_bbox((gx1, gy1, gx2, gy2), w, h)
            if gbox is None:
                tr["last_lp"] = frame_idx
                continue
            gx1, gy1, gx2, gy2 = gbox
            tr["last_best_lp_conf"] = float(plate.confidence)
            skip_vehicle_ocr = self._can_reuse_locked_ocr(
                tr,
                (float(gx1), float(gy1), float(gx2), float(gy2)),
            )
            if not skip_vehicle_ocr:
                ocr_crops.append(plate_crop)
                ocr_owner.append(
                    (tid, frame_idx, float(plate.confidence), (gx1, gy1, gx2, gy2))
                )
            else:
                times.ocr_locked_reuse_skips += 1
                label_str = (
                    f"{str(tr.get('ocr_text', '')).strip()} "
                    f"{float(tr.get('ocr_conf', 0.0)):.2f}"
                )
                draw_by_tid[tid] = [
                    (float(gx1), float(gy1), float(gx2), float(gy2), label_str)
                ]
                tr["plate_box"] = (float(gx1), float(gy1), float(gx2), float(gy2))
                tr["last_lp"] = frame_idx

        ocr_out: list[tuple[str, float]] = []
        if ocr_crops:
            ocr_st = OcrRecognizeStats()
            t_ocr = time.perf_counter()
            try:
                ocr_out = ocr.recognize(ocr_crops, stats=ocr_st)
            except Exception as e:
                console.print(f"[yellow]OCR error:[/] {str(e)}")
                ocr_out = [("", 0.0)] * len(ocr_crops)
            times.ocr_sec += time.perf_counter() - t_ocr
            times.ocr_recognize_calls += 1
            times.ocr_plates_submitted += len(ocr_crops)
            times.ocr_prefilter_skipped_plates += ocr_st.prefilter_skipped
            if len(ocr_out) != len(ocr_crops):
                ocr_out = list(ocr_out) + [("", 0.0)] * max(0, len(ocr_crops) - len(ocr_out))
                ocr_out = ocr_out[: len(ocr_crops)]

        for k, own in enumerate(ocr_owner):
            tid, frame_idx, lp_conf, gxy = own
            gx1, gy1, gx2, gy2 = gxy
            tr = self._tracks[tid]
            plate_text, ocr_conf = ocr_out[k] if k < len(ocr_out) else ("", 0.0)
            label_str = self._record_ocr_result(
                tr,
                lp_conf=lp_conf,
                plate_box=(float(gx1), float(gy1), float(gx2), float(gy2)),
                raw_text=str(plate_text),
                ocr_conf=float(ocr_conf),
            )
            draw_by_tid[tid] = [(float(gx1), float(gy1), float(gx2), float(gy2), label_str)]
            tr["last_lp"] = frame_idx

        for tid, rows in draw_by_tid.items():
            self._tracks[tid]["draw"] = rows

    def enqueue_lp_jobs_from_scene(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        frame_idx: int,
        lp_batch: LpBatchCoordinator | None,
    ) -> None:
        if lp_batch is None or not lp_batch.enabled:
            return
        h, w = frame.shape[:2]
        scene = _filter_scene_detections(detections)
        vehicles = [
            d for d in scene if d.label in VEHICLE_LABELS and d.confidence >= YOLO_CONFIDENCE
        ]
        veh_crops: list[tuple[int, int, int, int, np.ndarray, tuple[float, float, float, float]]] = []
        for v in vehicles:
            vb = clamp_bbox(v.bbox, w, h)
            if vb is None:
                continue
            vx1, vy1, vx2, vy2 = vb
            vehicle_crop = frame[vy1:vy2, vx1:vx2]
            if vehicle_crop.size == 0:
                continue
            fx = (float(vx1), float(vy1), float(vx2), float(vy2))
            veh_crops.append((vx1, vy1, vx2, vy2, vehicle_crop, fx))

        if not veh_crops:
            self._tracks.clear()
            return

        used_tids: set[int] = set()
        track_ids: list[int] = []
        for (_vx1, _vy1, _vx2, _vy2, _crop, fx) in veh_crops:
            best_tid: int | None = None
            best_iou = 0.0
            for tid, tr in self._tracks.items():
                if tid in used_tids:
                    continue
                iou = _iou_xyxy_f(fx, tr["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None and best_iou >= 0.3:
                tid = best_tid
                used_tids.add(tid)
                self._tracks[tid]["box"] = fx
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = self._new_track(fx)
            track_ids.append(tid)

        need_lp: list[int] = []
        for vi, tid in enumerate(track_ids):
            if self._need_lp(tid, frame_idx):
                need_lp.append(vi)

        for vi in need_lp:
            vx1, vy1, vx2, vy2, crop, _fx = veh_crops[vi]
            tid = track_ids[vi]
            lp_batch.enqueue_vehicle_crop(
                frame_idx=frame_idx,
                tid=tid,
                vx1=vx1,
                vy1=vy1,
                vx2=vx2,
                vy2=vy2,
                frame_w=w,
                frame_h=h,
                vehicle_crop=crop,
            )
        lp_batch.after_enqueue(frame_idx)

        stale = [tid for tid in self._tracks if tid not in track_ids]
        for tid in stale:
            del self._tracks[tid]

    def annotate(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        frame_idx: int,
        *,
        lp_detector: LpDetector,
        ocr: Ocr,
        annots: FrameAnnotators,
        times: PipelineStepTimes,
        run_scene_lp_ocr: bool = True,
        lp_inference: bool = True,
    ) -> None:
        h, w = frame.shape[:2]
        scene = _filter_scene_detections(detections)
        yolo_dets, yolo_labels = _detections_to_sv(scene, w, h)
        t_draw = time.perf_counter()
        if yolo_labels:
            annots.yolo_label.annotate(frame, yolo_dets, labels=yolo_labels)
        times.annotate_draw_sec += time.perf_counter() - t_draw

        if not run_scene_lp_ocr:
            plate_rows: list[list[float]] = []
            plate_confs: list[float] = []
            plate_label_strs: list[str] = []
            for _tid, tr in self._tracks.items():
                for gx1, gy1, gx2, gy2, label_str in tr.get("draw", []):
                    plate_rows.append([gx1, gy1, gx2, gy2])
                    plate_confs.append(1.0)
                    plate_label_strs.append(label_str)
            t_plate = time.perf_counter()
            if plate_rows:
                p_xyxy = np.asarray(plate_rows, dtype=np.float32)
                p_conf = np.asarray(plate_confs, dtype=np.float32)
                p_cls = np.zeros(len(plate_rows), dtype=np.int64)
                p_dets = sv.Detections(xyxy=p_xyxy, confidence=p_conf, class_id=p_cls)
                annots.plate_label.annotate(frame, p_dets, labels=plate_label_strs)
            times.annotate_draw_sec += time.perf_counter() - t_plate
            return

        vehicles = [
            d for d in scene if d.label in VEHICLE_LABELS and d.confidence >= YOLO_CONFIDENCE
        ]
        veh_crops: list[tuple[int, int, int, int, np.ndarray, tuple[float, float, float, float]]] = []
        for v in vehicles:
            vb = clamp_bbox(v.bbox, w, h)
            if vb is None:
                continue
            vx1, vy1, vx2, vy2 = vb
            vehicle_crop = frame[vy1:vy2, vx1:vx2]
            if vehicle_crop.size == 0:
                continue
            fx = (float(vx1), float(vy1), float(vx2), float(vy2))
            veh_crops.append((vx1, vy1, vx2, vy2, vehicle_crop, fx))

        if not veh_crops:
            self._tracks.clear()
            return

        used_tids: set[int] = set()
        track_ids: list[int] = []
        for (_vx1, _vy1, _vx2, _vy2, _crop, fx) in veh_crops:
            best_tid: int | None = None
            best_iou = 0.0
            for tid, tr in self._tracks.items():
                if tid in used_tids:
                    continue
                iou = _iou_xyxy_f(fx, tr["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None and best_iou >= 0.3:
                tid = best_tid
                used_tids.add(tid)
                self._tracks[tid]["box"] = fx
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = self._new_track(fx)
            track_ids.append(tid)

        need_lp: list[int] = []
        for vi, tid in enumerate(track_ids):
            if self._need_lp(tid, frame_idx):
                need_lp.append(vi)

        plate_rows: list[list[float]] = []
        plate_confs: list[float] = []
        plate_label_strs: list[str] = []

        draw_by_tid: dict[int, list[tuple[float, float, float, float, str]]] = {}

        if need_lp and lp_inference:
            fd_list = [
                FrameData(index=j, timestamp=0.0, image=veh_crops[vi][4])
                for j, vi in enumerate(need_lp)
            ]
            t_lp = time.perf_counter()
            plates_per_sub = lp_detector.detect_plates(fd_list)
            times.lp_sec += time.perf_counter() - t_lp

            ocr_crops: list[np.ndarray] = []
            ocr_owner: list[tuple[int, int, float, tuple[int, int, int, int]]] = []
            # (tid, vi_index_in_need_lp, lp_conf, global_xyxy)

            for j, vi in enumerate(need_lp):
                vx1, vy1, vx2, vy2, _crop, _fx = veh_crops[vi]
                tid = track_ids[vi]
                tr = self._tracks[tid]
                if j >= len(plates_per_sub):
                    tr["last_lp"] = frame_idx
                    continue
                best_plate: LicensePlate | None = None
                for plate in plates_per_sub[j]:
                    if plate.confidence < LP_CONFIDENCE:
                        continue
                    if best_plate is None or plate.confidence > best_plate.confidence:
                        best_plate = plate
                if best_plate is None:
                    tr["last_lp"] = frame_idx
                    continue
                plate = best_plate
                pbox = clamp_bbox(plate.bbox, veh_crops[vi][4].shape[1], veh_crops[vi][4].shape[0])
                if pbox is None:
                    tr["last_lp"] = frame_idx
                    continue
                px1, py1, px2, py2 = pbox
                plate_crop = veh_crops[vi][4][py1:py2, px1:px2]
                if plate_crop.size == 0:
                    tr["last_lp"] = frame_idx
                    continue
                gx1, gy1, gx2, gy2 = vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2
                gbox = clamp_bbox((gx1, gy1, gx2, gy2), w, h)
                if gbox is None:
                    tr["last_lp"] = frame_idx
                    continue
                gx1, gy1, gx2, gy2 = gbox
                tr["last_best_lp_conf"] = float(plate.confidence)
                skip_vehicle_ocr = self._can_reuse_locked_ocr(
                    tr,
                    (float(gx1), float(gy1), float(gx2), float(gy2)),
                )
                if not skip_vehicle_ocr:
                    ocr_crops.append(plate_crop)
                    ocr_owner.append(
                        (tid, j, float(plate.confidence), (gx1, gy1, gx2, gy2))
                    )
                else:
                    times.ocr_locked_reuse_skips += 1
                    label_str = (
                        f"{str(tr.get('ocr_text', '')).strip()} "
                        f"{float(tr.get('ocr_conf', 0.0)):.2f}"
                    )
                    draw_by_tid[tid] = [
                        (float(gx1), float(gy1), float(gx2), float(gy2), label_str)
                    ]
                    tr["plate_box"] = (float(gx1), float(gy1), float(gx2), float(gy2))
                    tr["last_lp"] = frame_idx

            ocr_out: list[tuple[str, float]] = []
            if ocr_crops:
                ocr_st = OcrRecognizeStats()
                t_ocr = time.perf_counter()
                try:
                    ocr_out = ocr.recognize(ocr_crops, stats=ocr_st)
                except Exception as e:
                    console.print(f"[yellow]OCR error:[/] {str(e)}")
                    ocr_out = [("", 0.0)] * len(ocr_crops)
                times.ocr_sec += time.perf_counter() - t_ocr
                times.ocr_recognize_calls += 1
                times.ocr_plates_submitted += len(ocr_crops)
                times.ocr_prefilter_skipped_plates += ocr_st.prefilter_skipped
                if len(ocr_out) != len(ocr_crops):
                    ocr_out = list(ocr_out) + [("", 0.0)] * max(0, len(ocr_crops) - len(ocr_out))
                    ocr_out = ocr_out[: len(ocr_crops)]

            for k, own in enumerate(ocr_owner):
                tid, _j, lp_conf, gxy = own
                gx1, gy1, gx2, gy2 = gxy
                tr = self._tracks[tid]
                plate_text, ocr_conf = ocr_out[k] if k < len(ocr_out) else ("", 0.0)
                label_str = self._record_ocr_result(
                    tr,
                    lp_conf=lp_conf,
                    plate_box=(float(gx1), float(gy1), float(gx2), float(gy2)),
                    raw_text=str(plate_text),
                    ocr_conf=float(ocr_conf),
                )
                draw_by_tid[tid] = [(float(gx1), float(gy1), float(gx2), float(gy2), label_str)]
                tr["last_lp"] = frame_idx

            for tid, rows in draw_by_tid.items():
                self._tracks[tid]["draw"] = rows

        stale = [tid for tid in self._tracks if tid not in track_ids]
        for tid in stale:
            del self._tracks[tid]

        for tid in track_ids:
            tr = self._tracks.get(tid)
            if not tr:
                continue
            for gx1, gy1, gx2, gy2, label_str in tr.get("draw", []):
                plate_rows.append([gx1, gy1, gx2, gy2])
                plate_confs.append(1.0)
                plate_label_strs.append(label_str)

        t_plate = time.perf_counter()
        if plate_rows:
            p_xyxy = np.asarray(plate_rows, dtype=np.float32)
            p_conf = np.asarray(plate_confs, dtype=np.float32)
            p_cls = np.zeros(len(plate_rows), dtype=np.int64)
            p_dets = sv.Detections(xyxy=p_xyxy, confidence=p_conf, class_id=p_cls)
            annots.plate_label.annotate(frame, p_dets, labels=plate_label_strs)
        times.annotate_draw_sec += time.perf_counter() - t_plate


@dataclass(frozen=True)
class FrameAnnotators:
    """Supervision annotators for boxes + labels on RF-DETR trash/cigarette."""

    trash_box: Any
    trash_label: Any
    yolo_label: Any
    plate_label: Any
    # Same scale/thickness passed into LabelAnnotator — reuse for peeing status line (cv2).
    label_text_scale: float
    label_text_thickness: int


def _make_frame_annotators(width: int, height: int) -> FrameAnnotators:
    """Trash/cigarette: colored boxes + labels; colors follow ``ColorLookup.INDEX`` on class_id."""
    wh = (int(width), int(height))
    base_thickness = int(sv.calculate_optimal_line_thickness(resolution_wh=wh))
    line_thickness = max(2, (base_thickness * 2 + 2) // 3)
    box_thickness = max(2, line_thickness)

    base_text_scale = float(sv.calculate_optimal_text_scale(resolution_wh=wh))
    text_scale = 0.5 * max(0.45, base_text_scale * 1.4)
    text_scale = max(0.22, float(text_scale))

    text_thickness = max(3, line_thickness + 2)
    lookup = sv.ColorLookup.INDEX
    sp = bool(ANNOTATOR_SMART_POSITION)

    trash_palette = sv.ColorPalette([sv.Color.RED, sv.Color.YELLOW])
    trash_box = sv.BoxAnnotator(
        color=trash_palette,
        thickness=box_thickness,
        color_lookup=lookup,
    )
    trash_label = sv.LabelAnnotator(
        color=trash_palette,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=text_thickness,
        smart_position=sp,
        color_lookup=lookup,
    )
    yolo_label = sv.LabelAnnotator(
        color=sv.Color.GREEN,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=text_thickness,
        smart_position=sp,
        color_lookup=lookup,
    )
    plate_label = sv.LabelAnnotator(
        color=sv.Color.BLUE,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=text_thickness,
        smart_position=sp,
        color_lookup=lookup,
    )
    return FrameAnnotators(
        trash_box=trash_box,
        trash_label=trash_label,
        yolo_label=yolo_label,
        plate_label=plate_label,
        label_text_scale=text_scale,
        label_text_thickness=text_thickness,
    )


def _trash_detections_to_sv(
    detections: Sequence[Detection], width: int, height: int
) -> tuple[sv.Detections, list[str]]:
    """RF-DETR labels: class_id 0 = trash (red), 1 = cigarette (yellow) for :class:`sv.BoxAnnotator`."""
    xyxy_list: list[list[float]] = []
    conf_list: list[float] = []
    cls_list: list[int] = []
    labels: list[str] = []
    for det in detections:
        try:
            x1, y1, x2, y2 = map(float, det.bbox)
        except Exception:
            continue
        bbox = clamp_bbox((int(x1), int(y1), int(x2), int(y2)), width, height)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        xyxy_list.append([float(x1), float(y1), float(x2), float(y2)])
        conf_list.append(float(det.confidence))
        lab_lower = str(det.label).lower()
        cls_list.append(1 if "cigarette" in lab_lower else 0)
        labels.append(f"{det.label} {det.confidence:.2f}")
    if not xyxy_list:
        empty = np.zeros((0, 4), dtype=np.float32)
        return sv.Detections(xyxy=empty), []
    xyxy = np.asarray(xyxy_list, dtype=np.float32)
    conf = np.asarray(conf_list, dtype=np.float32)
    class_id = np.asarray(cls_list, dtype=np.int64)
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id), labels


def _detections_to_sv(
    detections: Sequence[Detection], width: int, height: int
) -> tuple[sv.Detections, list[str]]:
    """Clamp boxes to frame and build supervision Detections + label strings."""
    xyxy_list: list[list[float]] = []
    conf_list: list[float] = []
    labels: list[str] = []
    for det in detections:
        try:
            x1, y1, x2, y2 = map(float, det.bbox)
        except Exception:
            continue
        bbox = clamp_bbox((int(x1), int(y1), int(x2), int(y2)), width, height)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        xyxy_list.append([float(x1), float(y1), float(x2), float(y2)])
        conf_list.append(float(det.confidence))
        labels.append(f"{det.label} {det.confidence:.2f}")
    if not xyxy_list:
        empty = np.zeros((0, 4), dtype=np.float32)
        return sv.Detections(xyxy=empty), []
    xyxy = np.asarray(xyxy_list, dtype=np.float32)
    conf = np.asarray(conf_list, dtype=np.float32)
    class_id = np.zeros(len(conf_list), dtype=np.int64)
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id), labels


def _load_trash_detector_required():
    """Load both RF-DETR heads from TensorRT ``.engine`` files only (no PyTorch / ONNX fallback)."""
    te = Path(TRASH_ENGINE_PATH)
    ce = Path(CIGARETTE_ENGINE_PATH)
    if not te.is_file():
        console.print(
            f"[red]RF-DETR requires trash TensorRT engine:[/] {te}\n"
            f"Set TRASH_ENGINE_PATH in settings.py (default: weights/trash_fp16_tensorRT.engine)."
        )
        raise SystemExit(2)
    if not ce.is_file():
        console.print(
            f"[red]RF-DETR requires cigarette TensorRT engine:[/] {ce}\n"
            f"Set CIGARETTE_ENGINE_PATH in settings.py (default: weights/cigarette_fp16_tensorRT.engine)."
        )
        raise SystemExit(2)
    if te.resolve() == ce.resolve():
        console.print("[red]Trash and cigarette engine paths must be two different files.[/]")
        raise SystemExit(2)
    from models.rfdetr_trt_trash import RfDetrTrtTrashDetector

    try:
        return RfDetrTrtTrashDetector(
            te,
            ce,
            class_names=None,
            conf_threshold=TRASH_CONFIDENCE,
        )
    except ImportError as exc:
        console.print(
            "[red]RF-DETR TensorRT requires tensorrt and pycuda.[/]\n"
            "Example: [bold]pip install tensorrt pycuda[/]"
        )
        raise SystemExit(2) from exc
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]RF-DETR TensorRT failed to load:[/] {exc}")
        raise SystemExit(3) from exc


def _scene_has_activity(detections: Sequence[Detection], min_conf: float) -> bool:
    """True if any person or vehicle detection meets confidence (YOLO class subset)."""
    for d in detections:
        if d.confidence < min_conf:
            continue
        if _is_scene_detection(d):
            return True
    return False


def _scene_has_vehicles_at_conf(detections: Sequence[Detection], min_conf: float) -> bool:
    """True if any road-vehicle class meets ``min_conf`` (used to gate LP/OCR work)."""
    for d in detections:
        if d.confidence < min_conf:
            continue
        if d.label in VEHICLE_LABELS:
            return True
    return False


def clamp_bbox(bbox, w, h):
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _draw_trash_detections(
    frame,
    trash_detections: Sequence[Detection],
    annots: FrameAnnotators,
) -> None:
    """Draw RF-DETR trash (red box) + cigarette (yellow box) with confidence labels."""
    h, w = frame.shape[:2]
    sv_dets, labels = _trash_detections_to_sv(trash_detections, w, h)
    if not labels:
        return
    annots.trash_box.annotate(frame, sv_dets)
    annots.trash_label.annotate(frame, sv_dets, labels=labels)


def _annotate_yolo_lp_ocr(
    frame,
    detections: Sequence[Detection],
    *,
    lp_detector: LpDetector,
    ocr: Ocr,
    annots: FrameAnnotators,
    times: PipelineStepTimes,
) -> None:
    """Draw YOLO labels (no boxes), then LP + OCR on vehicle crops (mutates ``frame`` in place).

    One **batched** LP inference for all vehicle crops in this frame, then one **batched**
    OCR call for all plate crops (reduces per-vehicle Python ↔ GPU overhead).
    """
    h, w = frame.shape[:2]
    scene = _filter_scene_detections(detections)

    yolo_dets, yolo_labels = _detections_to_sv(scene, w, h)
    t_draw = time.perf_counter()
    if yolo_labels:
        annots.yolo_label.annotate(frame, yolo_dets, labels=yolo_labels)
    times.annotate_draw_sec += time.perf_counter() - t_draw

    vehicles = [d for d in scene if d.label in VEHICLE_LABELS and d.confidence >= YOLO_CONFIDENCE]
    veh_crops: list[tuple[int, int, int, int, np.ndarray]] = []
    for v in vehicles:
        vb = clamp_bbox(v.bbox, w, h)
        if vb is None:
            continue
        vx1, vy1, vx2, vy2 = vb
        vehicle_crop = frame[vy1:vy2, vx1:vx2]
        if vehicle_crop.size == 0:
            continue
        veh_crops.append((vx1, vy1, vx2, vy2, vehicle_crop))

    if not veh_crops:
        return

    fd_list = [
        FrameData(index=i, timestamp=0.0, image=entry[4]) for i, entry in enumerate(veh_crops)
    ]
    t_lp = time.perf_counter()
    plates_per_frame = lp_detector.detect_plates(fd_list)
    times.lp_sec += time.perf_counter() - t_lp

    ocr_crops: list[np.ndarray] = []
    ocr_geos: list[tuple[int, int, int, int]] = []

    for i, (vx1, vy1, vx2, vy2, vehicle_crop) in enumerate(veh_crops):
        if i >= len(plates_per_frame):
            break
        for plate in plates_per_frame[i]:
            pbox = clamp_bbox(plate.bbox, vehicle_crop.shape[1], vehicle_crop.shape[0])
            if pbox is None:
                continue
            px1, py1, px2, py2 = pbox
            plate_crop = vehicle_crop[py1:py2, px1:px2]
            if plate_crop.size == 0:
                continue
            gx1, gy1, gx2, gy2 = vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2
            gbox = clamp_bbox((gx1, gy1, gx2, gy2), w, h)
            if gbox is None:
                continue
            gx1, gy1, gx2, gy2 = gbox
            ocr_crops.append(plate_crop)
            ocr_geos.append((gx1, gy1, gx2, gy2))

    if not ocr_crops:
        return

    t_ocr = time.perf_counter()
    try:
        ocr_st = OcrRecognizeStats()
        ocr_out = ocr.recognize(ocr_crops, stats=ocr_st)
    except Exception as e:
        console.print(f"[yellow]OCR error:[/] {str(e)}")
        ocr_out = [("", 0.0)] * len(ocr_crops)
        ocr_st = OcrRecognizeStats()
    times.ocr_sec += time.perf_counter() - t_ocr
    times.ocr_recognize_calls += 1
    times.ocr_plates_submitted += len(ocr_crops)
    times.ocr_prefilter_skipped_plates += ocr_st.prefilter_skipped
    if len(ocr_out) != len(ocr_crops):
        ocr_out = list(ocr_out) + [("", 0.0)] * max(0, len(ocr_crops) - len(ocr_out))
        ocr_out = ocr_out[: len(ocr_crops)]

    plate_rows: list[list[float]] = []
    plate_confs: list[float] = []
    plate_label_strs: list[str] = []
    for (gx1, gy1, gx2, gy2), ocr_one in zip(ocr_geos, ocr_out):
        plate_text, plate_conf = ocr_one
        plate_text = normalize_plate_text(plate_text)
        if plate_conf < PLATE_CONFIDENCE:
            continue
        plate_rows.append([float(gx1), float(gy1), float(gx2), float(gy2)])
        plate_confs.append(float(plate_conf))
        label_str = f"{plate_text} {plate_conf:.2f}" if plate_text else f"{plate_conf:.2f}"
        plate_label_strs.append(label_str)

    t_plate = time.perf_counter()
    if plate_rows:
        p_xyxy = np.asarray(plate_rows, dtype=np.float32)
        p_conf = np.asarray(plate_confs, dtype=np.float32)
        p_cls = np.zeros(len(plate_rows), dtype=np.int64)
        p_dets = sv.Detections(xyxy=p_xyxy, confidence=p_conf, class_id=p_cls)
        annots.plate_label.annotate(frame, p_dets, labels=plate_label_strs)
    times.annotate_draw_sec += time.perf_counter() - t_plate


def _annotate_frame(
    frame,
    trash_detections: Sequence[Detection],
    yolo_detections: Sequence[Detection],
    *,
    lp_detector: LpDetector,
    ocr: Ocr,
    annots: FrameAnnotators,
    peeing_state: PeeingState,
    times: PipelineStepTimes,
    frame_idx: int,
    lp_cache: VehicleLpOcrCache | None = None,
    run_scene_lp_ocr: bool = True,
    lp_inference: bool = True,
) -> None:
    t0 = time.perf_counter()
    _draw_trash_detections(frame, trash_detections, annots)
    times.annotate_draw_sec += time.perf_counter() - t0
    if lp_cache is not None:
        lp_cache.annotate(
            frame,
            yolo_detections,
            frame_idx,
            lp_detector=lp_detector,
            ocr=ocr,
            annots=annots,
            times=times,
            run_scene_lp_ocr=run_scene_lp_ocr,
            lp_inference=lp_inference,
        )
    else:
        _annotate_yolo_lp_ocr(
            frame,
            yolo_detections,
            lp_detector=lp_detector,
            ocr=ocr,
            annots=annots,
            times=times,
        )
    t0 = time.perf_counter()
    _draw_peeing_overlay(frame, peeing_state)
    times.annotate_draw_sec += time.perf_counter() - t0


def _rfdetr_engine_batch(trash: TrashDetector) -> int:
    """TensorRT engines use a static batch; PyTorch path may omit ``engine_batch_size``."""
    bs = getattr(trash, "engine_batch_size", None)
    return max(1, int(bs)) if bs is not None else 8


def _pad_rfdetr_frame(template: FrameData) -> FrameData:
    """Black frame for TensorRT fixed-batch padding (tail / streak break)."""
    blank = np.zeros_like(template.image)
    return FrameData(index=-1, timestamp=0.0, image=blank)


def _run_pipeline_uniform_stride_batched(
    *,
    cap: cv2.VideoCapture,
    out: VideoWriterSink,
    fps: float,
    total_frames: int,
    yolo: YoloDetector,
    lp_detector: LpDetector,
    ocr: Ocr,
    trash: TrashDetector,
    times: PipelineStepTimes,
    annots: FrameAnnotators,
    peeing: PeeingDetector,
    lp_cache: VehicleLpOcrCache | None,
    stride: int,
    lp_batch: LpBatchCoordinator | None = None,
) -> None:
    """Scene YOLO only on ``frame_idx % stride == 0``, micro-batched.

    Peeing still uses **carried** scene boxes between samples. LP/OCR and scene YOLO labels for
    plates run only on sampled frames where scene YOLO reports a vehicle at ``YOLO_CONFIDENCE``;
    other frames redraw **cached** plate text from :class:`VehicleLpOcrCache` without new LP/OCR calls.
    RF-DETR runs only on sampled frames with person/vehicle activity (same threshold as scene YOLO).
    """
    B = _rfdetr_engine_batch(trash)
    max_queue_latency = max(0, int(RF_DETR_MAX_QUEUE_LATENCY_FRAMES))
    ymb = max(1, int(YOLO_MICRO_BATCH_SIZE))
    window_size = max(stride * ymb, ymb)

    pbar = tqdm(total=total_frames, desc=f"Processing video (stride={stride})")
    frame_idx = 0
    emit_idx = 0
    stash: dict[int, tuple[np.ndarray, List[Detection], List[Detection], PeeingState]] = {}
    rfdetr_q: list[tuple[FrameData, List[Detection], PeeingState]] = []
    carry_peeing: List[Detection] = []

    def emit_ready() -> None:
        nonlocal emit_idx
        while emit_idx in stash:
            if lp_batch is not None and lp_batch.enabled and lp_cache is not None:
                lp_batch.flush_until_frame_ready(emit_idx)
            img, td, scene, pst = stash.pop(emit_idx)
            lp_run = _scene_has_vehicles_at_conf(scene, YOLO_CONFIDENCE)
            lp_infer = not (lp_batch is not None and lp_batch.enabled and lp_run)
            _annotate_frame(
                img,
                td,
                scene,
                lp_detector=lp_detector,
                ocr=ocr,
                annots=annots,
                peeing_state=pst,
                times=times,
                frame_idx=emit_idx,
                lp_cache=lp_cache,
                run_scene_lp_ocr=lp_run,
                lp_inference=lp_infer,
            )
            t0 = time.perf_counter()
            out.write(img)
            times.video_write_sec += time.perf_counter() - t0
            emit_idx += 1

    def flush_one_rfdetr_batch() -> None:
        nonlocal rfdetr_q
        if len(rfdetr_q) < B:
            return
        batch = rfdetr_q[:B]
        rfdetr_q = rfdetr_q[B:]
        fds = [x[0] for x in batch]
        times.rfdetr_trt_batches += 1
        times.rfdetr_input_frames += B
        t0 = time.perf_counter()
        outs = trash.detect_trash(fds)
        times.trash_sec += time.perf_counter() - t0
        for j in range(B):
            fd, scene, pst = batch[j]
            td = list(outs[j]) if j < len(outs) else []
            stash[fd.index] = (fd.image, td, scene, pst)
            if lp_cache is not None:
                lp_cache.enqueue_lp_jobs_from_scene(fd.image, scene, fd.index, lp_batch)
        emit_ready()

    def flush_rfdetr_padded_tail() -> None:
        nonlocal rfdetr_q
        n = len(rfdetr_q)
        if n == 0:
            return
        batch = list(rfdetr_q)
        rfdetr_q.clear()
        fds = [b[0] for b in batch]
        pad_fd = _pad_rfdetr_frame(fds[-1])
        while len(fds) < B:
            fds.append(pad_fd)
        times.rfdetr_trt_batches += 1
        times.rfdetr_input_frames += n
        times.rfdetr_trt_padded_slots += max(0, B - n)
        t0 = time.perf_counter()
        outs = trash.detect_trash(fds)
        times.trash_sec += time.perf_counter() - t0
        for j in range(n):
            fd, scene, pst = batch[j]
            td = list(outs[j]) if j < len(outs) else []
            stash[fd.index] = (fd.image, td, scene, pst)
            if lp_cache is not None:
                lp_cache.enqueue_lp_jobs_from_scene(fd.image, scene, fd.index, lp_batch)
        emit_ready()

    def maybe_flush_rfdetr_latency(anchor_frame_idx: int) -> None:
        if max_queue_latency <= 0 or not rfdetr_q:
            return
        oldest = rfdetr_q[0][0].index
        if anchor_frame_idx - oldest >= max_queue_latency:
            flush_rfdetr_padded_tail()

    try:
        while True:
            window: List[tuple[int, np.ndarray]] = []
            t_read = time.perf_counter()
            while len(window) < window_size:
                ret, frame = cap.read()
                if not ret:
                    break
                window.append((frame_idx, frame))
                frame_idx += 1
            times.other_sec += time.perf_counter() - t_read
            if not window:
                break

            sampled_fds = [
                FrameData(index=i, timestamp=i / fps, image=img.copy())
                for (i, img) in window
                if (i % stride) == 0
            ]
            scene_by_idx: dict[int, List[Detection]] = {}
            if sampled_fds:
                t_y = time.perf_counter()
                for s0 in range(0, len(sampled_fds), ymb):
                    sub = sampled_fds[s0 : s0 + ymb]
                    raw_lists = yolo.detect(sub)
                    for bi, fd in enumerate(sub):
                        lst = raw_lists[bi] if bi < len(raw_lists) else []
                        scene_by_idx[fd.index] = _filter_scene_detections(list(lst))
                times.yolo_sec += time.perf_counter() - t_y

            pose_prefetch_by_idx: dict[int, list[bool | None]] = {}
            if PEEING_YOLO_POSE_CROSS_FRAME_BATCH and sampled_fds:
                pref_items = [
                    (i, img, scene_by_idx[i])
                    for (i, img) in window
                    if (i % stride) == 0 and i in scene_by_idx
                ]
                if pref_items:
                    pose_prefetch_by_idx = peeing.prefetch_yolo_pose_hits_for_window(
                        pref_items,
                        YOLO_CONFIDENCE,
                    )

            for i, img in window:
                if i in scene_by_idx:
                    carry_peeing = list(scene_by_idx[i])
                scene_peeing = list(carry_peeing)
                run_yolo = (i % stride) == 0
                scene_for_stash = list(scene_by_idx[i]) if i in scene_by_idx else []
                fd_opt: FrameData | None = None
                if run_yolo:
                    fd_opt = FrameData(index=i, timestamp=i / fps, image=img.copy())

                t_p = time.perf_counter()
                ts = i / fps
                pre_hits: list[bool | None] | None = None
                if run_yolo and PEEING_YOLO_POSE_CROSS_FRAME_BATCH:
                    pre_hits = pose_prefetch_by_idx.get(i)
                pstate = peeing.update(
                    img,
                    scene_peeing,
                    run_yolo=run_yolo,
                    yolo_conf=YOLO_CONFIDENCE,
                    timestamp_sec=ts,
                    precomputed_yolo_pose_hits=pre_hits,
                )
                times.peeing_sec += time.perf_counter() - t_p
                if pstate.edge_enter:
                    console.print(f"[bold magenta]PEEING[/] frame={i}")
                if pstate.edge_exit:
                    console.print(f"[dim]PEEING off[/] frame={i}")

                if run_yolo and fd_opt is not None:
                    if _scene_has_activity(scene_for_stash, YOLO_CONFIDENCE):
                        rfdetr_q.append((fd_opt, scene_for_stash, pstate))
                        while len(rfdetr_q) >= B:
                            flush_one_rfdetr_batch()
                    else:
                        stash[i] = (img.copy(), [], scene_for_stash, pstate)
                        if lp_cache is not None:
                            lp_cache.enqueue_lp_jobs_from_scene(img, scene_for_stash, i, lp_batch)
                        maybe_flush_rfdetr_latency(i)
                        emit_ready()
                else:
                    stash[i] = (img.copy(), [], [], pstate)
                    maybe_flush_rfdetr_latency(i)
                    emit_ready()

                pbar.update(1)

        while len(rfdetr_q) >= B:
            flush_one_rfdetr_batch()
        flush_rfdetr_padded_tail()

        if lp_batch is not None and lp_batch.enabled:
            lp_batch.eof_flush()
            times.lp_coordinator_batches = lp_batch.lp_queue_flushes
            times.lp_coordinator_latency_events = lp_batch.lp_latency_flushes
            times.lp_coordinator_emit_barriers = lp_batch.lp_emit_flushes
            times.lp_coordinator_queue_full_flushes = lp_batch.lp_flush_queue_full
            times.lp_coordinator_eof_rounds = lp_batch.lp_flush_eof_rounds

        pbar.close()
    finally:
        try:
            pbar.close()
        except Exception:
            pass


if __name__ == "__main__":
    run_pipeline(VIDEO_PATH, OUTPUT_VIDEO)
