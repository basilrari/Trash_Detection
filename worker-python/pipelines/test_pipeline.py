#!/usr/bin/env python3
"""
YOLO → RF-DETR trash → LP → OCR on a local video; writes an annotated MP4.
LP and OCR are **batched per frame** across all vehicle crops (fewer small GPU calls).
Labels only (no bounding boxes) for trash, YOLO, and plates; YOLO is person + road vehicles.

Run from worker-python/ (put source videos in inputs/, results under outputs/ by default):

  python worker.py
  python worker.py inputs/myvideo.mp4
  python worker.py inputs/myvideo.mp4 -o outputs/custom.mp4
  python -m pipelines.test_pipeline   # uses paths from settings.py

**Gate (``GATE_MODE``)** — default ``yolo``; see ``settings.py`` and ``Readme.md`` § Gating.
RF-DETR runs only on frames where YOLO reports a **person or vehicle** at ``YOLO_CONFIDENCE``
(scene activity), not on every decoded frame.

**Output video** — default ``OUTPUT_VIDEO_ENCODER=auto``: use ``ffmpeg`` ``h264_nvenc`` when
available, else OpenCV ``mp4v``. Override with ``nvenc`` (fail if unusable) or ``mp4v``. Tune
``NVENC_PRESET`` / ``NVENC_CQ`` in ``settings.py`` (env vars).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
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
from core.types import Detection, FrameData
from core.yolo_stride_gate import YoloStrideGate, YoloStrideGateConfig
from settings import (
    CHUNK_SECONDS,
    CIGARETTE_ENGINE_PATH,
    FFMPEG_PATH,
    GATE_MODE,
    NVENC_CQ,
    NVENC_PRESET,
    OUTPUT_VIDEO,
    OUTPUT_VIDEO_ENCODER,
    PEEING_CROP_MARGIN,
    PEEING_GROIN_DIST_MAX,
    PEEING_GROIN_LOOSE_FACTOR,
    PEEING_ALARM_ENTER_HIT_FRACTION,
    PEEING_ALARM_EXIT_HIT_FRACTION,
    PEEING_ALARM_MIN_SAMPLES,
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
    PLATE_CONFIDENCE,
    TRASH_CONFIDENCE,
    TRASH_ENGINE_PATH,
    VIDEO_PATH,
    YOLO_COARSE_STRIDE,
    YOLO_CONFIDENCE,
    YOLO_DENSE_STRIDE,
    YOLO_DENSE_IDLE_MISS_STREAK,
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


def _log_pipeline_run_configuration(
    *,
    mode: str,
    video_path: str,
    width: int,
    height: int,
    fps: float,
    total_frames: int,
    chunk_frames: int,
    output_video: str,
    sink_label: str,
    trash: TrashDetector,
) -> None:
    """Gates, thresholds, TRT layout, preprocess, encoder, timing env flags."""
    te = Path(TRASH_ENGINE_PATH).resolve()
    ce = Path(CIGARETTE_ENGINE_PATH).resolve()
    rf_pre = "unknown"
    b0 = b1 = b2 = "?"
    heads = getattr(trash, "_heads", None)
    if heads:
        w0 = heads[0][0]
        rf_pre = (
            "PyTorch CUDA → TRT D2D input"
            if getattr(w0, "_want_cuda_preprocess", False)
            else "NumPy + OpenCV CPU → TRT H2D"
        )
        b0, b1, b2 = (getattr(w0, "batch", "?"), getattr(w0, "height", "?"), getattr(w0, "width", "?"))
    trt_pre_env = os.environ.get("RF_DETR_PREPROCESS_CUDA", "(unset)")
    trt_tim_env = os.environ.get("RF_DETR_TRT_TIMING", "(unset)")
    console.print("[bold]Run configuration[/]")
    console.print(f"  Video  [dim]{video_path}[/]  →  {width}×{height} @ {fps:.3f} fps, {total_frames} frames")
    if mode == "yolo":
        console.print(
            f"  Gate   [bold]GATE_MODE=yolo[/]  coarse={YOLO_COARSE_STRIDE}  dense={YOLO_DENSE_STRIDE}  "
            f"dense_idle_miss={YOLO_DENSE_IDLE_MISS_STREAK}  "
            f"RF-DETR only on YOLO frames with person/vehicle ≥{YOLO_CONFIDENCE}"
        )
    else:
        console.print(
            f"  Gate   [bold]GATE_MODE=off[/]  CHUNK_SECONDS={CHUNK_SECONDS}  →  {chunk_frames} frames/chunk  "
            f"RF-DETR only on chunk frames with person/vehicle ≥{YOLO_CONFIDENCE}"
        )
    console.print(
        f"  Conf   YOLO={YOLO_CONFIDENCE}  trash_RF={TRASH_CONFIDENCE}  plate={PLATE_CONFIDENCE}"
    )
    console.print(f"  TRT    static batch={b0}  input {b1}×{b2}  preprocess: [cyan]{rf_pre}[/]")
    console.print(f"         trash.engine   [dim]{te}[/]")
    console.print(f"         cigarette      [dim]{ce}[/]")
    console.print(
        f"  Env    RF_DETR_PREPROCESS_CUDA={trt_pre_env!r}  RF_DETR_TRT_TIMING={trt_tim_env!r}"
    )
    console.print(
        f"  Encode OUTPUT_VIDEO_ENCODER={OUTPUT_VIDEO_ENCODER!r}  "
        f"NVENC_PRESET={NVENC_PRESET!r}  NVENC_CQ={NVENC_CQ}  FFMPEG_PATH={FFMPEG_PATH!r}"
    )
    console.print(f"  Output [dim]{Path(output_video).resolve()}[/]")
    console.print(f"  Writer [cyan]{sink_label}[/]")
    console.print(f"  Peeing [dim]{Path(PEEING_POSE_MODEL_PATH).expanduser().resolve()}[/]")
    if _trt_timing_enabled():
        console.print("  [yellow]RF_DETR_TRT_TIMING is on[/] — expect extra [TRT] lines per batch.")


class VideoWriterSink(Protocol):
    """Common surface for OpenCV ``VideoWriter`` and ffmpeg-backed writers."""

    def write(self, frame: np.ndarray) -> None: ...

    def release(self) -> None: ...


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
    peeing_sec: float = 0.0
    annotate_draw_sec: float = 0.0
    lp_sec: float = 0.0
    ocr_sec: float = 0.0
    video_write_sec: float = 0.0
    other_sec: float = 0.0  # open video, build writer, gate setup, chunk assembly

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
        console.print(f"  RF-DETR (trash):      {self.trash_sec:8.2f} s")
        if self.rfdetr_input_frames > 0 and self.trash_sec > 0:
            eff = self.rfdetr_input_frames / self.trash_sec
            console.print(
                f"  [dim]RF-DETR inputs:   {self.rfdetr_input_frames:8d} frames "
                f"→ {eff:5.1f} eff. FPS (inputs ÷ RF-DETR time only)[/]"
            )
            console.print(
                "  [dim]RF-DETR note:[/] ``[RF-DETR] … fps`` logs are per ``detect_trash`` call; "
                "eff. FPS above is the fair average. CPU preprocess per batch plus two TRT "
                "forwards (trash + cigarette) dominate when ``[TRT]`` preprocess ms is high."
            )
        console.print(f"  Peeing (MediaPipe):   {self.peeing_sec:8.2f} s")
        console.print(f"  Annotate (draw only): {self.annotate_draw_sec:8.2f} s")
        console.print(f"  LP detect:            {self.lp_sec:8.2f} s")
        console.print(f"  OCR:                  {self.ocr_sec:8.2f} s")
        console.print(f"  Video write:          {self.video_write_sec:8.2f} s")
        console.print(f"  [dim]Sum (inference): {inf:8.2f} s[/]")
        console.print(f"  [bold]Wall clock total:   {wall_total_sec:8.2f} s[/]")


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


@dataclass(frozen=True)
class FrameAnnotators:
    """Supervision label stack (text only; boxes are not drawn)."""

    trash_label: Any
    yolo_label: Any
    plate_label: Any
    # Same scale/thickness passed into LabelAnnotator — reuse for peeing status line (cv2).
    label_text_scale: float
    label_text_thickness: int


def _make_frame_annotators(width: int, height: int) -> FrameAnnotators:
    """Label sizing from supervision heuristics; trash head style (red) for trash + cigarette."""
    wh = (int(width), int(height))
    base_thickness = int(sv.calculate_optimal_line_thickness(resolution_wh=wh))
    line_thickness = max(2, (base_thickness * 2 + 2) // 3)

    base_text_scale = float(sv.calculate_optimal_text_scale(resolution_wh=wh))
    text_scale = 0.5 * max(0.45, base_text_scale * 1.4)
    text_scale = max(0.22, float(text_scale))

    text_thickness = max(3, line_thickness + 2)
    lookup = sv.ColorLookup.INDEX

    trash_label = sv.LabelAnnotator(
        color=sv.Color.RED,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=text_thickness,
        smart_position=True,
        color_lookup=lookup,
    )
    yolo_label = sv.LabelAnnotator(
        color=sv.Color.GREEN,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=text_thickness,
        smart_position=True,
        color_lookup=lookup,
    )
    plate_label = sv.LabelAnnotator(
        color=sv.Color.BLUE,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=text_thickness,
        smart_position=True,
        color_lookup=lookup,
    )
    return FrameAnnotators(
        trash_label=trash_label,
        yolo_label=yolo_label,
        plate_label=plate_label,
        label_text_scale=text_scale,
        label_text_thickness=text_thickness,
    )


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
            "Place ``trash.engine`` under worker-python/weights/ or set TRASH_ENGINE_PATH."
        )
        raise SystemExit(2)
    if not ce.is_file():
        console.print(
            f"[red]RF-DETR requires cigarette TensorRT engine:[/] {ce}\n"
            "Place ``cigarette.engine`` under worker-python/weights/ or set CIGARETTE_ENGINE_PATH."
        )
        raise SystemExit(2)
    if te.resolve() == ce.resolve():
        console.print("[red]trash.engine and cigarette.engine must be two different files.[/]")
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
    """Draw RF-DETR trash + cigarette in red (same trash head style)."""
    h, w = frame.shape[:2]
    sv_dets, labels = _detections_to_sv(trash_detections, w, h)
    if not labels:
        return
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


def _draw_peeing_overlay(
    frame: np.ndarray,
    state: PeeingState,
    *,
    text_scale: float,
    text_thickness: int,
) -> None:
    """Top-left single line: algorithmic status (not human verification)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = float(max(0.22, min(0.95, text_scale)))
    thick = max(1, min(3, int(round(text_thickness * 0.55))))

    tier = state.status
    tier_word = {"confirmed": "CONFIRMED", "suspected": "SUSPECTED"}.get(
        tier, "SUSPECTED"
    )
    text = f"PEEING {tier_word}  |  window {state.score:.0%}  (auto)"
    colors = {
        "confirmed": ((50, 255, 255), (0, 0, 0)),
        "suspected": ((60, 180, 255), (0, 0, 0)),
    }
    fill, outline = colors.get(tier, colors["suspected"])

    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    pad_x, pad_y = 8, 6
    ox = 10
    top = 8
    baseline = top + pad_y + th
    left = ox - pad_x
    right = ox + tw + pad_x
    box_top = top
    box_bottom = int(baseline + bl + pad_y)

    overlay = frame.copy()
    cv2.rectangle(overlay, (left, box_top), (right, box_bottom), (24, 24, 24), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (left, box_top), (right, box_bottom), (80, 80, 80), 1)

    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(
            frame,
            text,
            (ox + dx, baseline + dy),
            font,
            scale,
            outline,
            thick + 1,
            cv2.LINE_AA,
        )
    cv2.putText(frame, text, (ox, baseline), font, scale, fill, thick, cv2.LINE_AA)


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
) -> None:
    t0 = time.perf_counter()
    _draw_trash_detections(frame, trash_detections, annots)
    times.annotate_draw_sec += time.perf_counter() - t0
    _annotate_yolo_lp_ocr(
        frame,
        yolo_detections,
        lp_detector=lp_detector,
        ocr=ocr,
        annots=annots,
        times=times,
    )
    t0 = time.perf_counter()
    _draw_peeing_overlay(
        frame,
        peeing_state,
        text_scale=annots.label_text_scale,
        text_thickness=annots.label_text_thickness,
    )
    times.annotate_draw_sec += time.perf_counter() - t0


def _detect_trash_scene_activity_frames_only(
    trash: TrashDetector,
    chunk_frames_list: List[FrameData],
    detections_per_frame: List[List[Detection]],
    times: PipelineStepTimes,
) -> List[List[Detection]]:
    """Run RF-DETR only on frames with person/vehicle at ``YOLO_CONFIDENCE``; others get ``[]``."""
    n = len(chunk_frames_list)
    trash_out: List[List[Detection]] = [[] for _ in range(n)]
    active_frames: List[FrameData] = []
    active_orig: List[int] = []
    for i, fd in enumerate(chunk_frames_list):
        scene = _filter_scene_detections(detections_per_frame[i])
        if _scene_has_activity(scene, YOLO_CONFIDENCE):
            active_frames.append(fd)
            active_orig.append(i)
    if not active_frames:
        return trash_out
    B = _rfdetr_engine_batch(trash)
    for start in range(0, len(active_frames), B):
        batch = active_frames[start : start + B]
        t0 = time.perf_counter()
        part = trash.detect_trash(batch)
        times.trash_sec += time.perf_counter() - t0
        times.rfdetr_input_frames += len(batch)
        for j in range(len(batch)):
            oi = active_orig[start + j]
            trash_out[oi] = list(part[j]) if j < len(part) else []
    return trash_out


def _run_pipeline_chunked(
    *,
    cap: cv2.VideoCapture,
    out: VideoWriterSink,
    fps: float,
    total_frames: int,
    chunk_frames: int,
    yolo: YoloDetector,
    lp_detector: LpDetector,
    ocr: Ocr,
    trash: TrashDetector,
    times: PipelineStepTimes,
    annots: FrameAnnotators,
    peeing: PeeingDetector,
) -> None:
    pbar = tqdm(total=total_frames, desc="Processing video")
    frame_idx = 0
    try:
        while True:
            chunk_frames_list: List[FrameData] = []
            t_read = time.perf_counter()
            for _ in range(chunk_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                timestamp = frame_idx / fps
                chunk_frames_list.append(FrameData(index=frame_idx, timestamp=timestamp, image=frame))
                frame_idx += 1
                pbar.update(1)
            times.other_sec += time.perf_counter() - t_read

            if not chunk_frames_list:
                break

            t0 = time.perf_counter()
            detections_per_frame = yolo.detect(chunk_frames_list)
            times.yolo_sec += time.perf_counter() - t0
            trash_per_frame = _detect_trash_scene_activity_frames_only(
                trash, chunk_frames_list, detections_per_frame, times
            )

            n = min(len(chunk_frames_list), len(detections_per_frame), len(trash_per_frame))
            if n == 0:
                continue

            for i in range(n):
                frame_data = chunk_frames_list[i]
                frame = frame_data.image
                scene_dets = _filter_scene_detections(detections_per_frame[i])
                t_p = time.perf_counter()
                pstate = peeing.update(
                    frame,
                    scene_dets,
                    run_yolo=True,
                    yolo_conf=YOLO_CONFIDENCE,
                    timestamp_sec=frame_data.timestamp,
                )
                times.peeing_sec += time.perf_counter() - t_p
                if pstate.edge_enter:
                    console.print(f"[bold magenta]PEEING[/] frame={frame_data.index}")
                if pstate.edge_exit:
                    console.print(f"[dim]PEEING off[/] frame={frame_data.index}")
                _annotate_frame(
                    frame,
                    trash_per_frame[i],
                    scene_dets,
                    lp_detector=lp_detector,
                    ocr=ocr,
                    annots=annots,
                    peeing_state=pstate,
                    times=times,
                )
                t0 = time.perf_counter()
                out.write(frame)
                times.video_write_sec += time.perf_counter() - t0

        pbar.close()
    finally:
        try:
            pbar.close()
        except Exception:
            pass


def _rfdetr_engine_batch(trash: TrashDetector) -> int:
    """TensorRT engines use a static batch; PyTorch path may omit ``engine_batch_size``."""
    bs = getattr(trash, "engine_batch_size", None)
    return max(1, int(bs)) if bs is not None else 8


def _pad_rfdetr_frame(template: FrameData) -> FrameData:
    """Black frame for TensorRT fixed-batch padding (tail / streak break)."""
    blank = np.zeros_like(template.image)
    return FrameData(index=-1, timestamp=0.0, image=blank)


def _run_pipeline_yolo_gated(
    *,
    cap: cv2.VideoCapture,
    out: VideoWriterSink,
    fps: float,
    total_frames: int,
    yolo: YoloDetector,
    lp_detector: LpDetector,
    ocr: Ocr,
    gate: YoloStrideGate,
    trash: TrashDetector,
    times: PipelineStepTimes,
    annots: FrameAnnotators,
    peeing: PeeingDetector,
) -> None:
    """YOLO stride gate with **batched** RF-DETR on consecutive **scene-active** YOLO frames.

    When ``run_yolo`` is true and YOLO sees a person or vehicle above ``YOLO_CONFIDENCE``,
    the frame is queued for RF-DETR; idle YOLO frames drain any partial queue (padded) then
    emit with empty trash. TensorRT batch size is ``B`` (engine static batch).
    """
    B = _rfdetr_engine_batch(trash)
    pbar = tqdm(total=total_frames, desc="Processing video (YOLO stride gate)")
    frame_idx = 0
    emit_idx = 0
    stash: dict[int, tuple[np.ndarray, List[Detection], List[Detection], PeeingState]] = {}
    rfdetr_q: list[tuple[FrameData, List[Detection], PeeingState]] = []

    def emit_ready() -> None:
        nonlocal emit_idx
        while emit_idx in stash:
            img, td, scene, pst = stash.pop(emit_idx)
            _annotate_frame(
                img,
                td,
                scene,
                lp_detector=lp_detector,
                ocr=ocr,
                annots=annots,
                peeing_state=pst,
                times=times,
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
        t0 = time.perf_counter()
        times.rfdetr_input_frames += B
        outs = trash.detect_trash(fds)
        times.trash_sec += time.perf_counter() - t0
        for j in range(B):
            fd, scene, pst = batch[j]
            td = list(outs[j]) if j < len(outs) else []
            stash[fd.index] = (fd.image, td, scene, pst)
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
        t0 = time.perf_counter()
        times.rfdetr_input_frames += n
        outs = trash.detect_trash(fds)
        times.trash_sec += time.perf_counter() - t0
        for j in range(n):
            fd, scene, pst = batch[j]
            td = list(outs[j]) if j < len(outs) else []
            stash[fd.index] = (fd.image, td, scene, pst)
        emit_ready()

    def drain_rfdetr_before_non_yolo() -> None:
        while len(rfdetr_q) >= B:
            flush_one_rfdetr_batch()
        flush_rfdetr_padded_tail()

    try:
        while True:
            t_read = time.perf_counter()
            ret, frame = cap.read()
            times.other_sec += time.perf_counter() - t_read
            if not ret:
                break

            run_yolo = gate.should_run_yolo(frame_idx)
            scene_dets: List[Detection] = []
            fd_opt: FrameData | None = None
            if run_yolo:
                fd_opt = FrameData(index=frame_idx, timestamp=frame_idx / fps, image=frame.copy())
                t0 = time.perf_counter()
                dets_list = yolo.detect([fd_opt])
                times.yolo_sec += time.perf_counter() - t0
                detections = dets_list[0] if dets_list else []
                scene_dets = _filter_scene_detections(detections)
                gate.observe(frame_idx, _scene_has_activity(scene_dets, YOLO_CONFIDENCE))

            t_p = time.perf_counter()
            ts = frame_idx / fps
            pstate = peeing.update(
                frame,
                scene_dets,
                run_yolo=run_yolo,
                yolo_conf=YOLO_CONFIDENCE,
                timestamp_sec=ts,
            )
            times.peeing_sec += time.perf_counter() - t_p
            if pstate.edge_enter:
                console.print(f"[bold magenta]PEEING[/] frame={frame_idx}")
            if pstate.edge_exit:
                console.print(f"[dim]PEEING off[/] frame={frame_idx}")

            if run_yolo and fd_opt is not None:
                if _scene_has_activity(scene_dets, YOLO_CONFIDENCE):
                    rfdetr_q.append((fd_opt, scene_dets, pstate))
                    while len(rfdetr_q) >= B:
                        flush_one_rfdetr_batch()
                else:
                    drain_rfdetr_before_non_yolo()
                    stash[frame_idx] = (frame.copy(), [], scene_dets, pstate)
                    emit_ready()
            else:
                drain_rfdetr_before_non_yolo()
                stash[frame_idx] = (frame.copy(), [], scene_dets, pstate)
                emit_ready()

            frame_idx += 1
            pbar.update(1)

        while len(rfdetr_q) >= B:
            flush_one_rfdetr_batch()
        flush_rfdetr_padded_tail()

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

    mode = GATE_MODE if GATE_MODE in ("off", "yolo") else "yolo"
    if mode != GATE_MODE:
        console.print(f"[yellow]Unknown GATE_MODE={GATE_MODE!r}; using 'yolo'[/]")

    _ensure_pytorch_cuda_kernels_work()
    _log_visible_torch_cuda_device()

    console.print("[bold]Models ready[/]")
    t0 = time.perf_counter()
    wdir = _worker_weights_dir()
    yolo = YoloDetector(conf_threshold=YOLO_CONFIDENCE)
    _log_model_ready("Scene YOLO", f"{(wdir / 'yolo11x.pt').resolve()} (Ultralytics, COCO person+vehicles)")
    lp_detector = LpDetector()
    _log_model_ready("License-plate YOLO", f"{(wdir / 'bestlicense.pt').resolve()} (Ultralytics)")
    ocr = Ocr()
    _log_model_ready("PaddleOCR", f"inference device={ocr.paddle_device}")
    trash = _load_trash_detector_required()
    heads = getattr(trash, "_heads", None)
    if heads:
        w0 = heads[0][0]
        _log_model_ready(
            "RF-DETR TensorRT",
            f"batch={w0.batch}  input {w0.height}×{w0.width}  "
            f"{'CUDA preprocess' if getattr(w0, '_want_cuda_preprocess', False) else 'CPU preprocess'}",
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

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        console.print(f"[red]Invalid FPS:[/] {fps}")
        sys.exit(4)

    chunk_frames = max(1, int(CHUNK_SECONDS * fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    console.print(
        f"[cyan]Video capture[/] opened  {width}×{height} @ {fps:.2f} fps  ({total_frames} frames)"
    )

    try:
        peeing = PeeingDetector(
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
    except Exception as exc:
        console.print(
            "[red]PeeingDetector failed to initialize (required).[/]\n"
            "Install MediaPipe in this environment, e.g. [bold]pip install mediapipe[/].\n"
            f"[dim]{exc}[/]"
        )
        raise SystemExit(2) from exc
    _log_model_ready(
        "PeeingDetector",
        f"MediaPipe pose [dim]{Path(PEEING_POSE_MODEL_PATH).expanduser().resolve()}[/]",
    )
    console.print(
        "[dim]Peeing hint:[/] standing + squat cues; straddle penalty; "
        f"alarm: last {PEEING_WINDOW_SEC:.0f}s of pose hits (score ≥{PEEING_POSE_MATCH_THRESHOLD:.0%}); "
        f"arm when >{PEEING_ALARM_ENTER_HIT_FRACTION:.0%} hits with ≥{PEEING_ALARM_MIN_SAMPLES} samples, "
        f"disarm when <{PEEING_ALARM_EXIT_HIT_FRACTION:.0%} (no per-person IDs)."
    )

    annots = _make_frame_annotators(width, height)

    out_dir = os.path.dirname(os.path.abspath(output_video))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out, sink_label = _open_output_video_sink(
        output_video,
        fps=float(fps),
        width=width,
        height=height,
    )
    _log_pipeline_run_configuration(
        mode=mode,
        video_path=video_path,
        width=width,
        height=height,
        fps=float(fps),
        total_frames=total_frames,
        chunk_frames=chunk_frames,
        output_video=output_video,
        sink_label=sink_label,
        trash=trash,
    )
    times.other_sec += time.perf_counter() - t_io

    try:
        if mode == "yolo":
            t_gate = time.perf_counter()
            stride_gate = YoloStrideGate(
                YoloStrideGateConfig(
                    coarse_stride=YOLO_COARSE_STRIDE,
                    dense_stride=YOLO_DENSE_STRIDE,
                    dense_idle_miss_streak=YOLO_DENSE_IDLE_MISS_STREAK,
                )
            )
            times.other_sec += time.perf_counter() - t_gate
            _run_pipeline_yolo_gated(
                cap=cap,
                out=out,
                fps=float(fps),
                total_frames=total_frames,
                yolo=yolo,
                lp_detector=lp_detector,
                ocr=ocr,
                gate=stride_gate,
                trash=trash,
                times=times,
                annots=annots,
                peeing=peeing,
            )
        else:
            _run_pipeline_chunked(
                cap=cap,
                out=out,
                fps=float(fps),
                total_frames=total_frames,
                chunk_frames=chunk_frames,
                yolo=yolo,
                lp_detector=lp_detector,
                ocr=ocr,
                trash=trash,
                times=times,
                annots=annots,
                peeing=peeing,
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
    times.print_summary(wall_total_sec=wall_total)


if __name__ == "__main__":
    run_pipeline(VIDEO_PATH, OUTPUT_VIDEO)
