#!/usr/bin/env python3
"""
Peeing-only video pipeline: scene YOLO (stride + micro-batch) + PeeingDetector + annotated MP4.

Reuses the same decode window, scene detection filtering, pose prefetch, and ``PeeingDetector.update``
semantics as the full ``pipelines.test_pipeline`` loop but does **not** load or run RF-DETR, LP, or OCR,
and does **not** import ``test_pipeline`` (lighter import graph: shared helpers live under ``pipelines/``).

Run from ``worker-python/``:

  python peeing_worker.py inputs/clip.mp4 -o outputs/clip_peeing_annotated.mp4
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from models.peeing_detector import PeeingDetector
from models.yolo_detector import YoloDetector
from models.types import Detection, FrameData
from pipelines.cuda_bootstrap import (
    _ensure_pytorch_cuda_kernels_work,
    _log_model_ready,
    _log_visible_torch_cuda_device,
    console,
)
from pipelines.frame_stride import _resolve_frame_sample_stride
from pipelines.peeing_shared import _draw_peeing_overlay, _filter_scene_detections
from pipelines.video_io import (
    NullVideoWriterSink,
    VideoWriterSink,
    _maybe_wrap_capture,
    _maybe_wrap_video_sink,
    _open_output_video_sink,
)
from settings import (
    FFMPEG_PATH,
    INPUT_VIDEO_FPS_MAX,
    INPUT_VIDEO_FPS_MIN,
    NVENC_CQ,
    NVENC_PRESET,
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
    YOLO_CONFIDENCE,
    YOLO_ENGINE_PATH,
    YOLO_MICRO_BATCH_SIZE,
    YOLO_TRT_BATCH_SIZE,
    YOLO_TRT_DYNAMIC,
    YOLO_TRT_IMAGE_SIZE,
)


def _format_clock_mm_ss(total_seconds: float) -> str:
    """Human-readable duration, e.g. ``45.12s`` or ``3m 12.50s``."""
    if total_seconds < 0:
        total_seconds = 0.0
    m = int(total_seconds // 60)
    s = total_seconds % 60.0
    if m == 0:
        return f"{s:.2f}s"
    return f"{m}m {s:.2f}s"


def _format_video_timestamp(sec: float) -> str:
    """Time position in video (H:MM:SS.mmm or M:SS.mmm if under 1h)."""
    if sec < 0:
        sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60.0
    if h > 0:
        return f"{h}:{m:02d}:{s:06.3f}"
    return f"{m}:{s:06.3f}"


@dataclass
class PeeingInterval:
    """One global confirmed-peeing active interval (``edge_enter`` … ``edge_exit`` or EOF)."""

    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    duration_sec: float
    start_time_str: str
    end_time_str: str
    duration_str: str
    n_confirmed_bboxes: int
    bboxes_json: str
    open_interval: bool


@dataclass
class PeeingOnlyStepTimes:
    """Cumulative seconds and counters for the peeing-only path."""

    init_sec: float = 0.0
    yolo_sec: float = 0.0
    yolo_input_frames: int = 0
    yolo_batch_launches: int = 0
    yolo_padded_slots: int = 0
    yolo_max_batch_slack: int = 0
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
    video_write_sec: float = 0.0
    other_sec: float = 0.0

    def print_summary(self, *, wall_total_sec: float) -> None:
        ann = self.annotate_draw_sec
        inf = self.yolo_sec + self.peeing_sec + ann + self.video_write_sec
        console.print("[bold]Step timings (peeing-only, cumulative)[/]")
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
        console.print(f"  Annotate (peeing only): {self.annotate_draw_sec:8.2f} s")
        console.print(f"  Video write:          {self.video_write_sec:8.2f} s")
        console.print(f"  [dim]Sum (inference + write in loop): {inf:8.2f} s[/]")
        console.print(f"  [bold]Wall clock total:   {wall_total_sec:8.2f} s[/]")


@dataclass
class PeeingOnlyVideoRecord:
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
    times: PeeingOnlyStepTimes | None = None
    events: list[PeeingInterval] = field(default_factory=list)


@dataclass
class PeeingPipelineOptions:
    """Runtime overrides for peeing-only runs (FN tuning, sweeps)."""

    hand_groin_y_threshold: float | None = None
    seconds_required: int | None = None
    collect_pose_viz: bool = False
    draw_pose: bool = False
    write_output: bool = True
    quiet: bool = False
    show_frame_progress: bool = True
    progress_label: str | None = None


@dataclass
class PeeingOnlyModelBundle:
    yolo: YoloDetector
    peeing: PeeingDetector
    init_sec: float
    options: PeeingPipelineOptions = field(default_factory=PeeingPipelineOptions)

    def cleanup(self) -> None:
        try:
            self.peeing.close()
        except Exception:
            pass


def _ingest_yolo_peeing_stats(
    times: PeeingOnlyStepTimes, yolo: YoloDetector, peeing: PeeingDetector
) -> None:
    ys = yolo.get_inference_batch_stats()
    times.yolo_input_frames = ys.input_units
    times.yolo_batch_launches = ys.batch_launches
    times.yolo_padded_slots = ys.padded_slots
    times.yolo_max_batch_slack = ys.max_batch_slack
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


def load_peeing_only_models(
    *,
    options: PeeingPipelineOptions | None = None,
) -> PeeingOnlyModelBundle:
    """Load scene YOLO and PeeingDetector only (no RF-DETR, LP, OCR)."""
    opts = options or PeeingPipelineOptions()
    hand_thr = (
        float(opts.hand_groin_y_threshold)
        if opts.hand_groin_y_threshold is not None
        else float(PEEING_HAND_GROIN_Y_THRESHOLD)
    )
    confirm_sec = (
        int(opts.seconds_required)
        if opts.seconds_required is not None
        else int(PEEING_SECONDS_REQUIRED)
    )

    _ensure_pytorch_cuda_kernels_work()
    if not opts.quiet:
        _log_visible_torch_cuda_device()

    if opts.quiet:
        console.print("[bold]Loading peeing-only models…[/]")
    else:
        console.print("[bold]Models ready (peeing-only)[/]")
    t0 = time.perf_counter()
    yolo = YoloDetector(conf_threshold=YOLO_CONFIDENCE)
    yolo_ready = (
        f"TensorRT [dim]{Path(YOLO_ENGINE_PATH).resolve()}[/]  "
        f"max_batch={YOLO_TRT_BATCH_SIZE} dynamic={YOLO_TRT_DYNAMIC}"
    )
    if not opts.quiet:
        _log_model_ready("Scene YOLO", yolo_ready)
    try:
        peeing = PeeingDetector(
            crop_margin=PEEING_CROP_MARGIN,
            min_visibility=PEEING_MIN_VISIBILITY,
            hand_groin_y_threshold=hand_thr,
            collect_pose_viz=opts.collect_pose_viz or opts.draw_pose,
            persist_pose_viz=(
                PEEING_PERSIST_POSE_VIZ
                if (opts.collect_pose_viz or opts.draw_pose)
                else False
            ),
            min_hits_per_second=PEEING_MIN_HITS_PER_SECOND,
            seconds_required=confirm_sec,
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
    if not opts.quiet:
        _log_model_ready("PeeingDetector", _pee_ready)
        _pee_hint_extra = (
            f"{Path(PEEING_YOLO_POSE_MODEL).name}  batch={PEEING_YOLO_POSE_BATCH_SIZE}  imgsz={PEEING_YOLO_POSE_IMGSZ}"
        )
        if PEEING_YOLO_POSE_DEVICE:
            _pee_hint_extra += f"; device={PEEING_YOLO_POSE_DEVICE!r}"
        console.print(
            "[dim]Peeing hint:[/] standing + hand near groin; "
            f"≥{PEEING_MIN_HITS_PER_SECOND} sampled pose hits per calendar second for "
            f"{PEEING_SECONDS_REQUIRED} consecutive seconds; IoU person tracks; "
            f"TensorRT/default: [dim]{_pee_hint_extra}[/]."
        )
        console.print(
            f"[dim]Scene YOLO TensorRT[/]  [dim]{Path(YOLO_ENGINE_PATH).resolve()}[/]  "
            f"max_batch={YOLO_TRT_BATCH_SIZE}  dynamic={YOLO_TRT_DYNAMIC}  imgsz={YOLO_TRT_IMAGE_SIZE}"
        )
    init_sec = time.perf_counter() - t0
    if opts.quiet:
        console.print(f"[dim]Models ready in {init_sec:.2f}s[/]")
    else:
        console.print(f"[dim]Model init wall time: {init_sec:.2f}s[/]")
    return PeeingOnlyModelBundle(
        yolo=yolo, peeing=peeing, init_sec=init_sec, options=opts
    )


def _log_peeing_only_run_configuration(
    *,
    video_path: str,
    width: int,
    height: int,
    fps: float,
    total_frames: int,
    output_video: str,
    sink_label: str,
    frame_stride: int,
    stride_detail: str,
) -> None:
    ymb = max(1, int(YOLO_MICRO_BATCH_SIZE))
    win = max(frame_stride * ymb, ymb)
    console.print("[bold]Run configuration (peeing-only)[/]")
    console.print(f"  Video  [dim]{video_path}[/]  →  {width}×{height} @ {fps:.3f} fps, {total_frames} frames")
    console.print(
        f"  Input    nominal FPS [{INPUT_VIDEO_FPS_MIN}, {INPUT_VIDEO_FPS_MAX}] "
        "(warning only if container FPS is outside)"
    )
    console.print(
        f"  Gate   [bold]effective stride={frame_stride}[/]  uniform scene-YOLO every {frame_stride} decoded "
        f"frame(s); read windows ≤{win} frames; ``YOLO_MICRO_BATCH_SIZE={ymb}``; "
        "non-sampled frames reuse last scene boxes for peeing."
    )
    console.print(f"         [dim]{stride_detail}[/]")
    console.print(f"  Conf   YOLO={YOLO_CONFIDENCE}")
    console.print(
        f"  Cfg    PEEING_YOLO_POSE_TRT_TIMING={repr(PEEING_YOLO_POSE_TRT_TIMING)}  "
        f"PEEING_YOLO_POSE_CROSS_FRAME_BATCH={repr(PEEING_YOLO_POSE_CROSS_FRAME_BATCH)}  "
        f"PEEING_YOLO_POSE_PREFETCH_DEBUG={repr(PEEING_YOLO_POSE_PREFETCH_DEBUG)}"
    )
    console.print(
        f"  Encode OUTPUT_VIDEO_ENCODER={repr(OUTPUT_VIDEO_ENCODER)}  "
        f"NVENC_PRESET={repr(NVENC_PRESET)}  NVENC_CQ={NVENC_CQ}  FFMPEG_PATH={FFMPEG_PATH!r}"
    )
    console.print(f"  Output [dim]{Path(output_video).resolve()}[/]")
    console.print(f"  Writer [cyan]{sink_label}[/]")
    console.print(
        f"  Pipeline YOLO_MICRO_BATCH_SIZE={YOLO_MICRO_BATCH_SIZE}  "
        f"PIPELINE_READ_AHEAD_QUEUE_SIZE={PIPELINE_READ_AHEAD_QUEUE_SIZE}  "
        f"PIPELINE_WRITE_QUEUE_SIZE={PIPELINE_WRITE_QUEUE_SIZE}"
    )


def _peeing_interval_from_boxes(
    *,
    start_frame: int,
    end_frame: int,
    start_sec: float,
    end_sec: float,
    boxes: list[list[float]],
    open_interval: bool,
) -> PeeingInterval:
    dur = max(0.0, end_sec - start_sec)
    return PeeingInterval(
        start_frame=start_frame,
        end_frame=end_frame,
        start_sec=start_sec,
        end_sec=end_sec,
        duration_sec=dur,
        start_time_str=_format_video_timestamp(start_sec),
        end_time_str=_format_video_timestamp(end_sec),
        duration_str=_format_clock_mm_ss(dur),
        n_confirmed_bboxes=len(boxes),
        bboxes_json=json.dumps(boxes),
        open_interval=open_interval,
    )


def _run_peeing_only_uniform_stride(
    *,
    cap: cv2.VideoCapture,
    out: VideoWriterSink,
    fps: float,
    total_frames: int,
    yolo: YoloDetector,
    peeing: PeeingDetector,
    times: PeeingOnlyStepTimes,
    stride: int,
    pipeline_options: PeeingPipelineOptions,
) -> list[PeeingInterval]:
    hand_thr = (
        float(pipeline_options.hand_groin_y_threshold)
        if pipeline_options.hand_groin_y_threshold is not None
        else float(PEEING_HAND_GROIN_Y_THRESHOLD)
    )
    draw_pose = bool(pipeline_options.draw_pose)
    write_output = bool(pipeline_options.write_output)
    quiet = bool(pipeline_options.quiet)
    ymb = max(1, int(YOLO_MICRO_BATCH_SIZE))
    window_size = max(stride * ymb, ymb)
    frame_desc = pipeline_options.progress_label or f"Peeing-only (stride={stride})"
    pbar = tqdm(
        total=total_frames,
        desc=frame_desc,
        disable=not pipeline_options.show_frame_progress,
        leave=False,
    )
    frame_idx = 0
    carry_peeing: List[Detection] = []
    events: list[PeeingInterval] = []
    open_start: tuple[int, float] | None = None
    last_boxes: list[list[float]] = []
    last_i = -1
    last_ts = 0.0

    try:
        while True:
            window: list[tuple[int, np.ndarray]] = []
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
                last_i = i
                last_ts = i / fps
                if i in scene_by_idx:
                    carry_peeing = list(scene_by_idx[i])
                scene_peeing = list(carry_peeing)
                run_yolo = (i % stride) == 0
                pre_hits: list[bool | None] | None = None
                if run_yolo and PEEING_YOLO_POSE_CROSS_FRAME_BATCH:
                    pre_hits = pose_prefetch_by_idx.get(i)

                t_p = time.perf_counter()
                ts = i / fps
                pstate = peeing.update(
                    img,
                    scene_peeing,
                    run_yolo=run_yolo,
                    yolo_conf=YOLO_CONFIDENCE,
                    timestamp_sec=ts,
                    frame_index=i if run_yolo else None,
                    precomputed_yolo_pose_hits=pre_hits,
                )
                times.peeing_sec += time.perf_counter() - t_p

                if pstate.active and pstate.mark_bboxes:
                    last_boxes = [[float(x) for x in b] for b in pstate.mark_bboxes]

                if pstate.edge_exit and open_start is not None:
                    sf, ss = open_start
                    boxes = last_boxes if last_boxes else []
                    events.append(
                        _peeing_interval_from_boxes(
                            start_frame=sf,
                            end_frame=i,
                            start_sec=ss,
                            end_sec=ts,
                            boxes=boxes,
                            open_interval=False,
                        )
                    )
                    open_start = None
                    last_boxes = []

                if pstate.edge_enter:
                    open_start = (i, ts)
                    if not last_boxes and pstate.mark_bboxes:
                        last_boxes = [[float(x) for x in b] for b in pstate.mark_bboxes]
                    if not quiet:
                        console.print(f"[bold magenta]PEEING[/] frame={i}")
                if pstate.edge_exit and not quiet:
                    console.print(f"[dim]PEEING off[/] frame={i}")

                if draw_pose or write_output:
                    out_img = img.copy()
                    t_a = time.perf_counter()
                    _draw_peeing_overlay(
                        out_img,
                        pstate,
                        draw_pose=draw_pose,
                        hand_groin_y_threshold=hand_thr,
                        min_visibility=PEEING_MIN_VISIBILITY,
                    )
                    times.annotate_draw_sec += time.perf_counter() - t_a
                    if write_output:
                        t_w = time.perf_counter()
                        out.write(out_img)
                        times.video_write_sec += time.perf_counter() - t_w
                pbar.update(1)

        if open_start is not None and last_i >= 0:
            sf, ss = open_start
            boxes = last_boxes if last_boxes else []
            events.append(
                _peeing_interval_from_boxes(
                    start_frame=sf,
                    end_frame=last_i,
                    start_sec=ss,
                    end_sec=last_ts,
                    boxes=boxes,
                    open_interval=True,
                )
            )
        return events
    finally:
        try:
            pbar.close()
        except Exception:
            pass


def run_peeing_pipeline_video(
    bundle: PeeingOnlyModelBundle,
    video_path: str,
    output_video: str,
    *,
    per_video_times_init_sec: float,
    models_init_sec: float,
    abort_on_error: bool = True,
    pipeline_options: PeeingPipelineOptions | None = None,
) -> PeeingOnlyVideoRecord:
    opts = pipeline_options or bundle.options
    def _fail(
        msg: str,
        *,
        wall_sec: float = 0.0,
        duration_sec: float = 0.0,
        fps: float = 0.0,
        width: int = 0,
        height: int = 0,
        total_frames: int = 0,
    ) -> PeeingOnlyVideoRecord:
        console.print(f"[red]{msg}[/]")
        if abort_on_error:
            m = msg.lower()
            if "not found" in m:
                sys.exit(2)
            if "invalid fps" in m:
                sys.exit(4)
            sys.exit(3)
        return PeeingOnlyVideoRecord(
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
    times = PeeingOnlyStepTimes()
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

    if not opts.quiet and (
        fps < float(INPUT_VIDEO_FPS_MIN) or fps > float(INPUT_VIDEO_FPS_MAX)
    ):
        console.print(
            f"[yellow]Warning:[/] reported FPS {fps:.2f} is outside the nominal input range "
            f"[{INPUT_VIDEO_FPS_MIN}, {INPUT_VIDEO_FPS_MAX}] — timing/stride math assumes "
            f"{INPUT_VIDEO_FPS_MIN}–{INPUT_VIDEO_FPS_MAX} fps."
        )

    stride_n, stride_detail = _resolve_frame_sample_stride(fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = (total_frames / fps) if fps > 0 else 0.0
    if not opts.quiet:
        console.print(
            f"[cyan]Video capture[/] opened  {width}×{height} @ {fps:.2f} fps  ({total_frames} frames)"
        )

    cap = _maybe_wrap_capture(cap, queue_size=PIPELINE_READ_AHEAD_QUEUE_SIZE)
    bundle.peeing.reset()

    if opts.write_output:
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
    else:
        out = NullVideoWriterSink()
        sink_label = "null (no encode)"
    if not opts.quiet:
        _log_peeing_only_run_configuration(
            video_path=video_path,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            output_video=output_video,
            sink_label=sink_label,
            frame_stride=stride_n,
            stride_detail=stride_detail,
        )
    times.other_sec += time.perf_counter() - t_io

    bundle.yolo.reset_inference_batch_stats()
    bundle.peeing.reset_inference_batch_stats()

    events: list[PeeingInterval] = []
    proc_exc: BaseException | None = None
    try:
        try:
            events = _run_peeing_only_uniform_stride(
                cap=cap,
                out=out,
                fps=fps,
                total_frames=total_frames,
                yolo=bundle.yolo,
                peeing=bundle.peeing,
                times=times,
                stride=stride_n,
                pipeline_options=opts,
            )
            if opts.write_output:
                console.print(f"[green]Peeing-only annotated video saved:[/] {output_video}")
        except BaseException as exc:
            proc_exc = exc
            console.print(f"[red]Peeing-only pipeline error:[/] {exc}")
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
    _ingest_yolo_peeing_stats(times, bundle.yolo, bundle.peeing)
    inference_sec = times.yolo_sec + times.peeing_sec + times.annotate_draw_sec
    encode_io_sec = times.video_write_sec + times.other_sec

    if proc_exc is not None:
        return PeeingOnlyVideoRecord(
            input_path=video_path,
            output_path=output_video,
            success=False,
            error=f"{type(proc_exc).__name__}: {proc_exc}",
            duration_sec=duration_sec,
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            wall_sec=wall_sec,
            inference_sec=inference_sec,
            encode_io_sec=encode_io_sec,
            models_init_sec=models_init_sec,
            times=times,
            events=events,
        )

    return PeeingOnlyVideoRecord(
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
        inference_sec=inference_sec,
        encode_io_sec=encode_io_sec,
        models_init_sec=models_init_sec,
        times=times,
        events=events,
    )


def run_peeing_pipeline(
    video_path: str,
    output_video: str,
    *,
    options: PeeingPipelineOptions | None = None,
) -> PeeingOnlyVideoRecord:
    """Load peeing-only models, process one video, print timing summary."""

    wall_start = time.perf_counter()
    if not os.path.exists(video_path):
        console.print(f"[red]Video not found:[/] {video_path}")
        sys.exit(2)

    bundle = load_peeing_only_models(options=options)
    try:
        rec = run_peeing_pipeline_video(
            bundle,
            video_path,
            output_video,
            per_video_times_init_sec=bundle.init_sec,
            models_init_sec=bundle.init_sec,
            abort_on_error=True,
            pipeline_options=options,
        )
    finally:
        bundle.cleanup()

    wall_total = time.perf_counter() - wall_start
    if rec.times is not None:
        rec.times.print_summary(wall_total_sec=wall_total)
    return rec
