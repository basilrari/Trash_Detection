#!/usr/bin/env python3
"""
Peeing video pipeline: external scene detections (e.g. DFINE) + PeeingDetector (pose TRT) + optional MP4.

**No built-in scene detector.** Provide ``scene_detector(frame_index, frame_bgr, timestamp_sec)``
returning ``list[Detection]`` — see ``DFINE_INTEGRATION.md`` and ``models.detection_contract``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np
from tqdm import tqdm

from models.detection_contract import prepare_scene_detections
from models.peeing_detector import PeeingDetector
from models.types import Detection
from pipelines.cuda_bootstrap import (
    _ensure_pytorch_cuda_kernels_work,
    _log_model_ready,
    _log_visible_torch_cuda_device,
    console,
)
from pipelines.frame_stride import _resolve_frame_sample_stride
from pipelines.peeing_shared import _draw_peeing_overlay
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
    DETECTION_WINDOW_BATCH,
    PEEING_DETECTION_CONFIDENCE,
    PIPELINE_READ_AHEAD_QUEUE_SIZE,
    PIPELINE_WRITE_QUEUE_SIZE,
)

# Called on stride-sampled frames only; return person + motorcycle/motorbike detections (required).
SceneDetectorFn = Callable[[int, np.ndarray, float], Sequence[Detection]]


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
    scene_detect_sec: float = 0.0
    scene_detect_frames: int = 0
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
        inf = self.scene_detect_sec + self.peeing_sec + ann + self.video_write_sec
        console.print("[bold]Step timings (peeing kit, cumulative)[/]")
        console.print(f"  Model init:           {self.init_sec:8.2f} s")
        console.print(f"  Other (I/O, chunks):  {self.other_sec:8.2f} s")
        console.print(f"  D-FINE / scene detect: {self.scene_detect_sec:8.2f} s")
        if self.scene_detect_frames > 0 and self.scene_detect_sec > 0:
            eff = self.scene_detect_frames / self.scene_detect_sec
            console.print(
                f"  [dim]D-FINE sampled frames: {self.scene_detect_frames:8d} "
                f"→ {eff:5.1f} eff. FPS (frames ÷ scene_detect time only)[/]"
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
    # Host hook (e.g. DFINE boxes) drawn on each output frame before peeing overlay.
    pre_peeing_overlay_fn: Optional[Callable[[np.ndarray, int], None]] = None


@dataclass
class PeeingOnlyModelBundle:
    peeing: PeeingDetector
    init_sec: float
    options: PeeingPipelineOptions = field(default_factory=PeeingPipelineOptions)

    def cleanup(self) -> None:
        try:
            self.peeing.close()
        except Exception:
            pass


def _ingest_peeing_stats(times: PeeingOnlyStepTimes, peeing: PeeingDetector) -> None:
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


def load_peeing_models(
    *,
    options: PeeingPipelineOptions | None = None,
    yolo_pose_model: str | None = None,
) -> PeeingOnlyModelBundle:
    """Load PeeingDetector (YOLO pose TRT only). Scene boxes come from your detector."""
    opts = options or PeeingPipelineOptions()
    pose_weights = (
        str(yolo_pose_model).strip()
        if yolo_pose_model
        else str(PEEING_YOLO_POSE_MODEL)
    )
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
        console.print("[bold]Loading peeing models…[/]")
    else:
        console.print("[bold]Models ready (peeing kit)[/]")
    t0 = time.perf_counter()
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
            yolo_pose_model=pose_weights,
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
    _pee_ready = f"YOLO pose [dim]{Path(pose_weights).expanduser().resolve()}[/]"
    if not opts.quiet:
        _log_model_ready("PeeingDetector", _pee_ready)
        _pee_hint_extra = (
            f"{Path(pose_weights).name}  batch={PEEING_YOLO_POSE_BATCH_SIZE}  imgsz={PEEING_YOLO_POSE_IMGSZ}"
        )
        if PEEING_YOLO_POSE_DEVICE:
            _pee_hint_extra += f"; device={PEEING_YOLO_POSE_DEVICE!r}"
        console.print(
            "[dim]Peeing hint:[/] standing + hand near groin; "
            f"≥{PEEING_MIN_HITS_PER_SECOND} sampled pose hits per calendar second for "
            f"{confirm_sec} consecutive seconds; IoU person tracks; "
            f"person boxes from your scene detector (e.g. DFINE); "
            f"TensorRT pose: [dim]{_pee_hint_extra}[/]."
        )
    init_sec = time.perf_counter() - t0
    if opts.quiet:
        console.print(f"[dim]Models ready in {init_sec:.2f}s[/]")
    else:
        console.print(f"[dim]Model init wall time: {init_sec:.2f}s[/]")
    return PeeingOnlyModelBundle(peeing=peeing, init_sec=init_sec, options=opts)


load_peeing_only_models = load_peeing_models


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
    ymb = max(1, int(DETECTION_WINDOW_BATCH))
    win = max(frame_stride * ymb, ymb)
    console.print("[bold]Run configuration (peeing kit)[/]")
    console.print(f"  Video  [dim]{video_path}[/]  →  {width}×{height} @ {fps:.3f} fps, {total_frames} frames")
    console.print(
        f"  Input    nominal FPS [{INPUT_VIDEO_FPS_MIN}, {INPUT_VIDEO_FPS_MAX}] "
        "(warning only if container FPS is outside)"
    )
    console.print(
        f"  Gate   [bold]effective stride={frame_stride}[/]  external detector every {frame_stride} decoded "
        f"frame(s); read windows ≤{win} frames; non-sampled frames reuse last boxes."
    )
    console.print(f"         [dim]{stride_detail}[/]")
    console.print(f"  Conf   PEEING_DETECTION_CONFIDENCE={PEEING_DETECTION_CONFIDENCE}")
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
        f"  Pipeline DETECTION_WINDOW_BATCH={DETECTION_WINDOW_BATCH}  "
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
    scene_detector: SceneDetectorFn,
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
    det_conf = float(PEEING_DETECTION_CONFIDENCE)
    ymb = max(1, int(DETECTION_WINDOW_BATCH))
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

            sampled_items = [(i, img) for (i, img) in window if (i % stride) == 0]
            scene_by_idx: dict[int, List[Detection]] = {}
            if sampled_items:
                t_y = time.perf_counter()
                batch_fn = getattr(scene_detector, "detect_batch", None)
                if batch_fn is not None:
                    batch_items = [(i, img, i / fps) for i, img in sampled_items]
                    raw_by_idx = batch_fn(batch_items)
                    for i, raw in raw_by_idx.items():
                        scene_by_idx[i] = prepare_scene_detections(
                            raw,
                            min_confidence=det_conf,
                            keep_motorcycles=PEEING_MOTORCYCLE_EXCLUSION_ENABLED,
                        )
                        times.scene_detect_frames += 1
                else:
                    for i, img in sampled_items:
                        ts = i / fps
                        raw = scene_detector(i, img, ts)
                        scene_by_idx[i] = prepare_scene_detections(
                            raw,
                            min_confidence=det_conf,
                            keep_motorcycles=PEEING_MOTORCYCLE_EXCLUSION_ENABLED,
                        )
                        times.scene_detect_frames += 1
                times.scene_detect_sec += time.perf_counter() - t_y

            pose_prefetch_by_idx: dict[int, list[bool | None]] = {}
            if PEEING_YOLO_POSE_CROSS_FRAME_BATCH and sampled_items:
                pref_items = [
                    (i, img, scene_by_idx[i])
                    for (i, img) in window
                    if (i % stride) == 0 and i in scene_by_idx
                ]
                if pref_items:
                    pose_prefetch_by_idx = peeing.prefetch_yolo_pose_hits_for_window(
                        pref_items,
                        det_conf,
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
                    yolo_conf=det_conf,
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
                    if pipeline_options.pre_peeing_overlay_fn is not None:
                        pipeline_options.pre_peeing_overlay_fn(out_img, i)
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
    scene_detector: SceneDetectorFn,
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
                scene_detector=scene_detector,
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
    _ingest_peeing_stats(times, bundle.peeing)
    inference_sec = times.scene_detect_sec + times.peeing_sec + times.annotate_draw_sec
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
    scene_detector: SceneDetectorFn,
    options: PeeingPipelineOptions | None = None,
) -> PeeingOnlyVideoRecord:
    """Load peeing models, process one video with your ``scene_detector``, print timing."""

    wall_start = time.perf_counter()
    if not os.path.exists(video_path):
        console.print(f"[red]Video not found:[/] {video_path}")
        sys.exit(2)

    bundle = load_peeing_models(options=options)
    try:
        rec = run_peeing_pipeline_video(
            bundle,
            video_path,
            output_video,
            scene_detector=scene_detector,
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
