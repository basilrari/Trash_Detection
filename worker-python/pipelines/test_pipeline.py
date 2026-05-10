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

**Output video** — ``OUTPUT_VIDEO_ENCODER`` in ``settings.py`` (``auto``, ``nvenc``, ``mp4v``).
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Protocol, Sequence

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from rich.console import Console

from models.base import TrashDetector
from models.yolo_detector import YoloDetector
from models.lp_detector import LpDetector
from models.ocr import Ocr
from models.peeing_detector import PeeingDetector, PeeingState
from models.rfdetr_trt_trash import _trt_timing_enabled
from models.types import Detection, FrameData, LicensePlate
from pipelines.lp_batch_coordinator import LpBatchCoordinator, LpQueuedCrop
from settings import (
    ANNOTATOR_SMART_POSITION,
    CIGARETTE_ENGINE_PATH,
    FFMPEG_PATH,
    FRAME_SAMPLE_STRIDE_OVERRIDE,
    SCENE_YOLO_TARGET_FRAMES_PER_SECOND,
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
    PEEING_MEDIAPIPE_DELEGATE,
    PEEING_MEDIAPIPE_MODE,
    PEEING_MIN_HITS_PER_SECOND,
    PEEING_MIN_VISIBILITY,
    PEEING_POSE_BACKEND,
    PEEING_POSE_MODEL_PATH,
    PEEING_POSE_MODEL_URL,
    PEEING_SECONDS_REQUIRED,
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


def _ensure_pytorch_cuda_kernels_work() -> None:
    """Fail fast if PyTorch cannot execute on ``cuda:0`` (common on very new GPUs / wheel mismatch).

    Ultralytics (scene YOLO + license-plate YOLO) uses PyTorch CUDA. Without this check,
    a typical failure is ``cudaErrorNoKernelImageForDevice`` mid-run, then Paddle teardown aborts.
    """
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    try:
        x = torch.randn(32, 32, device="cuda", dtype=torch.float32)
        _ = x @ x
        torch.cuda.synchronize()
    except Exception as exc:
        cap = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        n = int(torch.cuda.device_count())
        if n > 1:
            multi = (
                "\n  [bold]If you have a second GPU[/] (e.g. Ampere/Ada), pin this process to it:\n"
                "    CUDA_VISIBLE_DEVICES=1 python worker.py ...\n"
            )
        else:
            multi = (
                "\n  Install a PyTorch build that includes kernels for this GPU "
                "(see https://pytorch.org/get-started/locally/ ), or use a supported GPU.\n"
            )
        console.print(
            "[red]PyTorch cannot run CUDA kernels on the current default GPU.[/]\n"
            f"  cuda:0  name={name!r}  capability=sm_{cap[0]}{cap[1]}\n"
            "  Scene YOLO and LP YOLO require working torch CUDA.\n"
            f"{multi}"
            f"  [dim]{type(exc).__name__}: {exc}[/]"
        )
        raise SystemExit(2) from exc


def _log_visible_torch_cuda_device() -> None:
    """Log which GPU this process uses as ``torch.cuda:0`` (after ``CUDA_VISIBLE_DEVICES`` remap)."""
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        console.print("[cyan]CUDA[/] torch: not available (CPU)")
        return
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
    console.print(
        f"[cyan]CUDA[/] CUDA_VISIBLE_DEVICES={vis}  →  torch cuda:0 = {name!r}  "
        f"capability sm_{cap[0]}{cap[1]}"
    )


def _worker_weights_dir() -> Path:
    """``worker-python/weights`` (same base as default YOLO/LP weights)."""
    return Path(__file__).resolve().parents[1] / "weights"


def _log_model_ready(title: str, detail: str) -> None:
    console.print(f"  [green]OK[/] [bold]{title}[/]  [dim]— {detail}[/]")


def _resolve_frame_sample_stride(fps: float) -> tuple[int, str]:
    """Pick decoded-frame stride: optional fixed override, else ~``fps/target`` frames sampled per second."""
    if FRAME_SAMPLE_STRIDE_OVERRIDE is not None:
        n = max(1, int(FRAME_SAMPLE_STRIDE_OVERRIDE))
        return n, f"fixed ``FRAME_SAMPLE_STRIDE_OVERRIDE={n}``"
    target = float(SCENE_YOLO_TARGET_FRAMES_PER_SECOND)
    if target <= 0:
        target = 5.0
    lo = float(INPUT_VIDEO_FPS_MIN)
    hi = float(INPUT_VIDEO_FPS_MAX)
    fps_clamped = min(max(float(fps), lo), hi)
    n = max(1, int(round(fps_clamped / target)))
    approx_per_sec = fps_clamped / float(n)
    return n, (
        f"automatic: reported FPS={fps:.3f}; clamp [{lo:.0f},{hi:.0f}] → {fps_clamped:.2f}; "
        f"``SCENE_YOLO_TARGET_FRAMES_PER_SECOND={target:g}`` → stride={n} "
        f"(~{approx_per_sec:.2f} scene-YOLO frames/s of video)"
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
        (
            f"{Path(PEEING_YOLO_POSE_MODEL).expanduser().resolve()}  "
            f"batch={PEEING_YOLO_POSE_BATCH_SIZE}  backend=yolo"
        )
        if PEEING_POSE_BACKEND == "yolo"
        else f"{Path(PEEING_POSE_MODEL_PATH).expanduser().resolve()}  backend=mediapipe"
    )
    console.print(f"  Peeing [dim]{_pee_cfg}[/]")
    if _trt_timing_enabled() or (
        PEEING_POSE_BACKEND == "yolo" and bool(PEEING_YOLO_POSE_TRT_TIMING)
    ):
        parts = []
        if _trt_timing_enabled():
            parts.append("RF-DETR ``[TRT]``")
        if PEEING_POSE_BACKEND == "yolo" and bool(PEEING_YOLO_POSE_TRT_TIMING):
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


class VideoWriterSink(Protocol):
    """Common surface for OpenCV ``VideoWriter`` and ffmpeg-backed writers."""

    def write(self, frame: np.ndarray) -> None: ...

    def release(self) -> None: ...


class _AsyncVideoWriterSink:
    """Bounded queue + background thread so NVENC/ffmpeg ``write`` does not stall inference."""

    def __init__(self, inner: VideoWriterSink, max_queue: int) -> None:
        self._inner = inner
        qsize = max(1, int(max_queue))
        self._q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=qsize)
        self._thr = threading.Thread(
            target=self._loop, name="pipeline-async-video-writer", daemon=True
        )
        self._thr.start()

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            self._inner.write(item)

    def write(self, frame: np.ndarray) -> None:
        self._q.put(frame)

    def release(self) -> None:
        self._q.put(None)
        self._thr.join(timeout=600.0)
        self._inner.release()


class _ReadAheadVideoCapture:
    """Decode thread with a bounded queue ahead of the processing loop."""

    def __init__(self, cap: cv2.VideoCapture, max_queue: int) -> None:
        self._cap = cap
        qsize = max(1, int(max_queue))
        self._q: queue.Queue[tuple[bool, np.ndarray] | None] = queue.Queue(maxsize=qsize)

        def worker() -> None:
            """Push ``(ret, frame)`` for every read; after EOF push ``None`` so ``read()`` never blocks forever."""
            try:
                while True:
                    ret, frame = self._cap.read()
                    self._q.put((ret, frame))
                    if not ret:
                        break
            finally:
                try:
                    self._q.put(None, timeout=5.0)
                except Exception:
                    try:
                        self._q.put_nowait(None)
                    except Exception:
                        pass

        self._stream_ended = False
        self._thr = threading.Thread(target=worker, name="pipeline-read-ahead", daemon=True)
        self._thr.start()

    def read(self) -> tuple[bool, np.ndarray]:
        if self._stream_ended:
            return False, np.array([], dtype=np.uint8)
        item = self._q.get()
        if item is None:
            self._stream_ended = True
            return False, np.array([], dtype=np.uint8)
        return item

    def get(self, prop: int) -> float:
        return float(self._cap.get(prop))

    def release(self) -> None:
        try:
            if self._thr.is_alive():
                self._thr.join(timeout=30.0)
        except Exception:
            pass
        try:
            self._cap.release()
        except Exception:
            pass

    def isOpened(self) -> bool:
        return bool(self._cap.isOpened())


def _maybe_wrap_video_sink(sink: VideoWriterSink, *, queue_size: int) -> VideoWriterSink:
    if queue_size <= 0:
        return sink
    return _AsyncVideoWriterSink(sink, queue_size)


def _maybe_wrap_capture(cap: cv2.VideoCapture, *, queue_size: int) -> cv2.VideoCapture | _ReadAheadVideoCapture:
    if queue_size <= 0:
        return cap
    return _ReadAheadVideoCapture(cap, queue_size)


def _ffmpeg_nvenc_smoke_test(ffmpeg_bin: str) -> bool:
    """Return True if ``ffmpeg`` can run one frame through ``h264_nvenc`` (driver + build).

    Uses 256×256 frames: NVENC rejects very small sizes (e.g. 16×16) with
    ``Frame Dimension less than the minimum supported value`` even when the encoder exists.
    """
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=256x256:d=0.04",
        "-frames:v",
        "1",
        "-c:v",
        "h264_nvenc",
        "-f",
        "null",
        "-",
    ]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            timeout=20,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    return r.returncode == 0


class _Cv2Mp4vSink:
    """OpenCV ``mp4v`` into MP4 (CPU MPEG-4 Part 2; portable fallback)."""

    def __init__(self, path: str, fps: float, width: int, height: int) -> None:
        self._w = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not self._w.isOpened():
            raise RuntimeError(f"OpenCV VideoWriter failed to open: {path!r}")

    def write(self, frame: np.ndarray) -> None:
        self._w.write(frame)

    def release(self) -> None:
        self._w.release()


class _FfmpegNvencSink:
    """Stream raw BGR frames into ``ffmpeg`` ``h264_nvenc`` (GPU encoder, usually much faster)."""

    def __init__(
        self,
        path: str,
        fps: float,
        width: int,
        height: int,
        *,
        ffmpeg_bin: str,
        preset: str,
        cq: int,
    ) -> None:
        self._width = int(width)
        self._height = int(height)
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{self._width}x{self._height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "h264_nvenc",
            "-preset",
            preset,
            "-rc",
            "vbr",
            "-cq",
            str(int(cq)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            path,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if self._proc.stdin is None:
            self._proc.kill()
            raise RuntimeError("ffmpeg did not provide a stdin pipe")

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[0] != self._height or frame.shape[1] != self._width:
            raise ValueError(
                f"Frame shape {frame.shape[:2]} does not match encoder {self._height}x{self._width}"
            )
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        assert self._proc.stdin is not None
        self._proc.stdin.write(frame.tobytes())

    def release(self) -> None:
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
            self._proc.stdin = None
        rc = self._proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited with code {rc} while finishing {self._proc.args!r}")


def _open_output_video_sink(
    output_path: str,
    *,
    fps: float,
    width: int,
    height: int,
    encoder_mode: str | None = None,
) -> tuple[VideoWriterSink, str]:
    """Open a video sink: NVENC (ffmpeg) when allowed and available, else OpenCV ``mp4v``."""
    mode = (encoder_mode or OUTPUT_VIDEO_ENCODER or "auto").strip().lower()
    p = FFMPEG_PATH.strip()
    ffmpeg_bin = shutil.which(p)
    if ffmpeg_bin is None and os.path.isabs(p) and os.path.isfile(p) and os.access(p, os.X_OK):
        ffmpeg_bin = p

    want_nvenc = mode in ("auto", "nvenc")
    nvenc_ok = bool(ffmpeg_bin and _ffmpeg_nvenc_smoke_test(ffmpeg_bin))

    if want_nvenc and nvenc_ok:
        try:
            sink: VideoWriterSink = _FfmpegNvencSink(
                output_path,
                fps,
                width,
                height,
                ffmpeg_bin=ffmpeg_bin,
                preset=NVENC_PRESET,
                cq=NVENC_CQ,
            )
            return (
                sink,
                f"ffmpeg h264_nvenc (preset={NVENC_PRESET!r}, cq={NVENC_CQ})",
            )
        except Exception as exc:
            if mode == "nvenc":
                console.print(
                    "[red]OUTPUT_VIDEO_ENCODER=nvenc but ffmpeg NVENC writer failed:[/]\n"
                    f"  [dim]{type(exc).__name__}: {exc}[/]"
                )
                raise SystemExit(3) from exc
            console.print(
                f"[yellow]ffmpeg h264_nvenc failed ({type(exc).__name__}: {exc}); "
                "falling back to OpenCV mp4v.[/]"
            )

    if mode == "nvenc" and not nvenc_ok:
        console.print(
            "[red]OUTPUT_VIDEO_ENCODER=nvenc but h264_nvenc is not usable "
            f"(ffmpeg={ffmpeg_bin!r}). Install ffmpeg with NVENC and a working NVIDIA driver.[/]"
        )
        raise SystemExit(3)

    if mode == "auto" and want_nvenc and not nvenc_ok:
        reason = (
            "ffmpeg not on PATH"
            if not ffmpeg_bin
            else "h264_nvenc smoke test failed (driver / ffmpeg build?)"
        )
        console.print(f"[dim]Video encoder:[/] {reason}; using OpenCV mp4v.")

    sink2: VideoWriterSink = _Cv2Mp4vSink(output_path, fps, width, height)
    return sink2, "OpenCV VideoWriter mp4v (CPU)"


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
                f"emit_barriers={self.lp_coordinator_emit_barriers}"
            )
        console.print(f"  OCR:                  {self.ocr_sec:8.2f} s")
        console.print(f"  Video write:          {self.video_write_sec:8.2f} s")
        console.print(f"  [dim]Sum (inference): {inf:8.2f} s[/]")
        console.print(f"  [bold]Wall clock total:   {wall_total_sec:8.2f} s[/]")


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


# Match Ultralytics COCO names for YOLO ``classes=[0,2,3,5,7]`` (person + road vehicles).
VEHICLE_LABELS = ("car", "truck", "bus", "motorcycle", "motorbike", "vehicle")
PERSON_LABELS = ("person",)


def _is_scene_detection(d: Detection) -> bool:
    """YOLO is restricted to person + vehicles; keep this aligned with ``YoloDetector.classes``."""
    return d.label in PERSON_LABELS or d.label in VEHICLE_LABELS


def _filter_scene_detections(detections: Sequence[Detection]) -> List[Detection]:
    return [d for d in detections if _is_scene_detection(d)]


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
            t_ocr = time.perf_counter()
            try:
                ocr_out = ocr.recognize(ocr_crops)
            except Exception as e:
                console.print(f"[yellow]OCR error:[/] {str(e)}")
                ocr_out = [("", 0.0)] * len(ocr_crops)
            times.ocr_sec += time.perf_counter() - t_ocr
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
                t_ocr = time.perf_counter()
                try:
                    ocr_out = ocr.recognize(ocr_crops)
                except Exception as e:
                    console.print(f"[yellow]OCR error:[/] {str(e)}")
                    ocr_out = [("", 0.0)] * len(ocr_crops)
                times.ocr_sec += time.perf_counter() - t_ocr
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
        ocr_out = ocr.recognize(ocr_crops)
    except Exception as e:
        console.print(f"[yellow]OCR error:[/] {str(e)}")
        ocr_out = [("", 0.0)] * len(ocr_crops)
    times.ocr_sec += time.perf_counter() - t_ocr
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


def _draw_peeing_overlay(frame: np.ndarray, state: PeeingState) -> None:
    """Thick red boxes only for confirmed peeing (latched), on sampled frames."""
    if not state.sampled:
        return
    red_bgr = (0, 0, 255)
    thick = 6
    for x1, y1, x2, y2 in state.mark_bboxes:
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(frame, p1, p2, red_bgr, thick)


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
            if (
                PEEING_POSE_BACKEND == "yolo"
                and PEEING_YOLO_POSE_CROSS_FRAME_BATCH
                and sampled_fds
            ):
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
                if (
                    run_yolo
                    and PEEING_POSE_BACKEND == "yolo"
                    and PEEING_YOLO_POSE_CROSS_FRAME_BATCH
                ):
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

        pbar.close()
    finally:
        try:
            pbar.close()
        except Exception:
            pass


def run_pipeline(video_path: str, output_video: str) -> None:
    """Process ``video_path`` and write annotated video to ``output_video``."""
    wall_start = time.perf_counter()
    times = PipelineStepTimes()

    if not os.path.exists(video_path):
        console.print(f"[red]Video not found:[/] {video_path}")
        sys.exit(2)

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
    t_init_done = time.perf_counter()
    times.init_sec = t_init_done - t0
    console.print(f"[dim]Model init wall time: {times.init_sec:.2f}s[/]")

    t_io = time.perf_counter()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        console.print(f"[red]Failed to open video:[/] {video_path}")
        sys.exit(3)

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        console.print(f"[red]Invalid FPS:[/] {fps}")
        sys.exit(4)

    if fps < float(INPUT_VIDEO_FPS_MIN) or fps > float(INPUT_VIDEO_FPS_MAX):
        console.print(
            f"[yellow]Warning:[/] reported FPS {fps:.2f} is outside the nominal input range "
            f"[{INPUT_VIDEO_FPS_MIN}, {INPUT_VIDEO_FPS_MAX}] — timing/stride math assumes {INPUT_VIDEO_FPS_MIN}–{INPUT_VIDEO_FPS_MAX} fps."
        )

    stride_n, stride_detail = _resolve_frame_sample_stride(fps)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    console.print(
        f"[cyan]Video capture[/] opened  {width}×{height} @ {fps:.2f} fps  ({total_frames} frames)"
    )

    cap = _maybe_wrap_capture(cap, queue_size=PIPELINE_READ_AHEAD_QUEUE_SIZE)

    try:
        peeing = PeeingDetector(
            crop_margin=PEEING_CROP_MARGIN,
            min_visibility=PEEING_MIN_VISIBILITY,
            hand_groin_y_threshold=PEEING_HAND_GROIN_Y_THRESHOLD,
            min_hits_per_second=PEEING_MIN_HITS_PER_SECOND,
            seconds_required=PEEING_SECONDS_REQUIRED,
            track_iou_threshold=PEEING_TRACK_IOU_THRESHOLD,
            track_max_missed_seconds=PEEING_TRACK_MAX_MISSED_SECONDS,
            pose_backend=PEEING_POSE_BACKEND,
            model_path=PEEING_POSE_MODEL_PATH,
            model_url=PEEING_POSE_MODEL_URL,
            mediapipe_mode=PEEING_MEDIAPIPE_MODE,
            mediapipe_delegate=PEEING_MEDIAPIPE_DELEGATE,
            yolo_pose_model=PEEING_YOLO_POSE_MODEL,
            yolo_pose_imgsz=PEEING_YOLO_POSE_IMGSZ,
            yolo_pose_batch_size=PEEING_YOLO_POSE_BATCH_SIZE,
            yolo_pose_trt_dynamic=PEEING_YOLO_POSE_TRT_DYNAMIC,
            yolo_pose_device=PEEING_YOLO_POSE_DEVICE,
            yolo_pose_trt_timing=PEEING_YOLO_POSE_TRT_TIMING,
            yolo_pose_prefetch_debug=PEEING_YOLO_POSE_PREFETCH_DEBUG,
            debug_timing=PEEING_DEBUG_TIMING,
            max_pose_persons_per_frame=PEEING_MAX_POSE_PERSONS_PER_FRAME,
        )
    except Exception as exc:
        console.print(
            "[red]PeeingDetector failed to initialize (required).[/]\n"
            "If pose_backend is [bold]yolo[/]: ensure ``pip install ultralytics torch`` and a pose "
            "``PEEING_YOLO_POSE_MODEL`` TensorRT ``.engine`` path.\n"
            "If pose_backend is [bold]mediapipe[/]: ``pip install mediapipe`` and the Tasks pose bundle.\n"
            f"[dim]{exc}[/]"
        )
        raise SystemExit(2) from exc
    _pee_ready = (
        f"YOLO pose [dim]{Path(PEEING_YOLO_POSE_MODEL).expanduser().resolve()}[/]"
        if PEEING_POSE_BACKEND == "yolo"
        else f"MediaPipe pose [dim]{Path(PEEING_POSE_MODEL_PATH).expanduser().resolve()}[/]"
    )
    _log_model_ready(
        "PeeingDetector",
        _pee_ready,
    )
    if PEEING_POSE_BACKEND == "yolo":
        _pee_hint_extra = (
            f"YOLO pose TensorRT/default: [dim]{Path(PEEING_YOLO_POSE_MODEL).name}[/]  "
            f"batch={PEEING_YOLO_POSE_BATCH_SIZE}  imgsz={PEEING_YOLO_POSE_IMGSZ}"
        )
        if PEEING_YOLO_POSE_DEVICE:
            _pee_hint_extra += f"; device={PEEING_YOLO_POSE_DEVICE!r}"
    else:
        _pee_hint_extra = (
            f"MediaPipe Tasks; mode [cyan]{PEEING_MEDIAPIPE_MODE}[/], "
            f"delegate [cyan]{PEEING_MEDIAPIPE_DELEGATE}[/]"
        )
    console.print(
        "[dim]Peeing hint:[/] standing + hand near groin; "
        f"≥{PEEING_MIN_HITS_PER_SECOND} sampled pose hits per calendar second for "
        f"{PEEING_SECONDS_REQUIRED} consecutive seconds; IoU person tracks; "
        f"{_pee_hint_extra}."
    )

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
            peeing.close()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        try:
            out.release()
        except Exception:
            pass

    wall_total = time.perf_counter() - wall_start
    _ingest_ultralytics_pipeline_stats(times, yolo, lp_detector, peeing)
    times.print_summary(wall_total_sec=wall_total)


if __name__ == "__main__":
    run_pipeline(VIDEO_PATH, OUTPUT_VIDEO)
