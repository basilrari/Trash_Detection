#!/usr/bin/env python3
"""
YOLO → RF-DETR trash → LP → OCR on a local video; writes an annotated MP4.

Run from worker-python/ (put source videos in inputs/, results under outputs/ by default):

  python worker.py
  python worker.py inputs/myvideo.mp4
  python worker.py inputs/myvideo.mp4 -o outputs/custom.mp4
  python -m pipelines.test_pipeline   # uses paths from settings.py

**Gate (``GATE_MODE``)** — default ``yolo``; see ``settings.py`` and ``Readme.md`` § Gating.

**RF-DETR** — required: ``pip install rfdetr`` and ``weights/trash.pth`` (optional ``cigarette.pth``).
See ``models/trash_detector.py``.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence

import cv2
from tqdm import tqdm
from rich.console import Console

from models.yolo_detector import YoloDetector
from models.lp_detector import LpDetector
from models.ocr import Ocr
from core.types import Detection, FrameData
from core.yolo_stride_gate import YoloStrideGate, YoloStrideGateConfig
from settings import (
    CHUNK_SECONDS,
    CIGARETTE_WEIGHTS_PATH,
    GATE_MODE,
    OUTPUT_VIDEO,
    PLATE_CONFIDENCE,
    RF_DETR_SIZE,
    TRASH_CONFIDENCE,
    TRASH_WEIGHTS_PATH,
    VIDEO_PATH,
    YOLO_COARSE_STRIDE,
    YOLO_CONFIDENCE,
    YOLO_DENSE_STRIDE,
    YOLO_DENSE_IDLE_MISS_STREAK,
)

if TYPE_CHECKING:
    from models.trash_detector import RfDetrTrashDetector

console = Console()


@dataclass
class PipelineStepTimes:
    """Cumulative seconds spent in each stage (wall-clock, ``perf_counter``)."""

    init_sec: float = 0.0
    yolo_sec: float = 0.0
    trash_sec: float = 0.0
    annotate_sec: float = 0.0
    video_write_sec: float = 0.0
    other_sec: float = 0.0  # open video, build writer, gate setup, chunk assembly

    def print_summary(self, *, wall_total_sec: float) -> None:
        inf = self.yolo_sec + self.trash_sec + self.annotate_sec + self.video_write_sec
        console.print("[bold]Step timings (cumulative)[/]")
        console.print(f"  Model init:           {self.init_sec:8.2f} s")
        console.print(f"  Other (I/O, chunks):  {self.other_sec:8.2f} s")
        console.print(f"  YOLO:                 {self.yolo_sec:8.2f} s")
        console.print(f"  RF-DETR (trash):      {self.trash_sec:8.2f} s")
        console.print(f"  Annotate + LP + OCR: {self.annotate_sec:8.2f} s")
        console.print(f"  Video write:          {self.video_write_sec:8.2f} s")
        console.print(f"  [dim]Sum (inference): {inf:8.2f} s[/]")
        console.print(f"  [bold]Wall clock total:   {wall_total_sec:8.2f} s[/]")


VEHICLE_LABELS = ("vehicle", "car", "truck", "bus", "motorbike", "motorcycle")
PERSON_LABELS = ("person",)


def _load_trash_detector_required() -> "RfDetrTrashDetector":
    """Load RF-DETR; exit the process if ``rfdetr`` or weights are missing."""
    tw = Path(TRASH_WEIGHTS_PATH)
    if not tw.is_file():
        console.print(
            f"[red]RF-DETR is required but weights file not found:[/] {tw}\n"
            f"Set TRASH_WEIGHTS_PATH or place trash.pth under worker-python/weights/."
        )
        raise SystemExit(2)
    from models.trash_detector import RfDetrTrashDetector

    cig = Path(CIGARETTE_WEIGHTS_PATH)
    cig_path: str | None = str(cig) if cig.is_file() else None
    try:
        return RfDetrTrashDetector(
            tw,
            cigarette_weights_path=cig_path,
            class_names=None,
            conf_threshold=TRASH_CONFIDENCE,
            model_size=RF_DETR_SIZE,
        )
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", "") == "rfdetr":
            console.print(
                "[red]RF-DETR is required but the ``rfdetr`` package is not installed.[/]\n"
                "Install with: [bold]pip install rfdetr[/]"
            )
            raise SystemExit(2) from exc
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]RF-DETR failed to load (required):[/] {exc}")
        raise SystemExit(3) from exc


def _scene_has_activity(detections: Sequence[Detection], min_conf: float) -> bool:
    """True if any person or vehicle-like detection meets confidence."""
    for d in detections:
        if d.confidence < min_conf:
            continue
        if d.label in VEHICLE_LABELS or d.label in PERSON_LABELS:
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


def _draw_trash_detections(frame, trash_detections: Sequence[Detection]) -> None:
    """Draw RF-DETR boxes in red (BGR) under YOLO."""
    h, w = frame.shape[:2]
    for det in trash_detections:
        try:
            x1, y1, x2, y2 = map(int, det.bbox)
        except Exception:
            continue
        bbox = clamp_bbox((x1, y1, x2, y2), w, h)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"{det.label} {det.confidence:.2f}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )


def _annotate_yolo_lp_ocr(
    frame,
    detections: Sequence[Detection],
    *,
    lp_detector: LpDetector,
    ocr: Ocr,
) -> None:
    """Draw YOLO boxes, then LP + OCR on vehicle crops (mutates ``frame`` in place)."""
    h, w = frame.shape[:2]

    for det in detections:
        try:
            x1, y1, x2, y2 = map(int, det.bbox)
        except Exception:
            continue
        bbox = clamp_bbox((x1, y1, x2, y2), w, h)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{det.label} {det.confidence:.2f}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    vehicles = [d for d in detections if d.label in VEHICLE_LABELS and d.confidence >= YOLO_CONFIDENCE]
    for v in vehicles:
        vb = clamp_bbox(v.bbox, w, h)
        if vb is None:
            continue
        vx1, vy1, vx2, vy2 = vb

        vehicle_crop = frame[vy1:vy2, vx1:vx2]
        if vehicle_crop.size == 0:
            continue

        plates_per_frame = lp_detector.detect_plates([FrameData(0, 0.0, vehicle_crop)])
        if not plates_per_frame:
            continue
        plates = plates_per_frame[0]

        for plate in plates:
            pbox = clamp_bbox(plate.bbox, vehicle_crop.shape[1], vehicle_crop.shape[0])
            if pbox is None:
                continue
            px1, py1, px2, py2 = pbox
            plate_crop = vehicle_crop[py1:py2, px1:px2]
            if plate_crop.size == 0:
                continue

            try:
                ocr_out = ocr.recognize([plate_crop])
            except Exception as e:
                console.print(f"[yellow]OCR error:[/] {str(e)}")
                ocr_out = [("", 0.0)]

            plate_text, plate_conf = ocr_out[0] if ocr_out else ("", 0.0)

            if plate_conf < PLATE_CONFIDENCE:
                continue

            cv2.rectangle(frame, (vx1 + px1, vy1 + py1), (vx1 + px2, vy1 + py2), (255, 0, 0), 2)
            label_str = f"{plate_text} {plate_conf:.2f}" if plate_text else f"{plate_conf:.2f}"
            cv2.putText(
                frame,
                label_str,
                (vx1 + px1, max(0, vy1 + py1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )


def _annotate_frame(
    frame,
    trash_detections: Sequence[Detection],
    yolo_detections: Sequence[Detection],
    *,
    lp_detector: LpDetector,
    ocr: Ocr,
) -> None:
    _draw_trash_detections(frame, trash_detections)
    _annotate_yolo_lp_ocr(frame, yolo_detections, lp_detector=lp_detector, ocr=ocr)


def _run_pipeline_chunked(
    *,
    cap: cv2.VideoCapture,
    out: cv2.VideoWriter,
    fps: float,
    total_frames: int,
    chunk_frames: int,
    yolo: YoloDetector,
    lp_detector: LpDetector,
    ocr: Ocr,
    trash: "RfDetrTrashDetector",
    times: PipelineStepTimes,
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
            t0 = time.perf_counter()
            trash_per_frame = trash.detect_trash(chunk_frames_list)
            times.trash_sec += time.perf_counter() - t0

            n = min(len(chunk_frames_list), len(detections_per_frame), len(trash_per_frame))
            if n == 0:
                continue

            for i in range(n):
                frame_data = chunk_frames_list[i]
                frame = frame_data.image
                t0 = time.perf_counter()
                _annotate_frame(
                    frame,
                    trash_per_frame[i],
                    detections_per_frame[i],
                    lp_detector=lp_detector,
                    ocr=ocr,
                )
                times.annotate_sec += time.perf_counter() - t0
                t0 = time.perf_counter()
                out.write(frame)
                times.video_write_sec += time.perf_counter() - t0

        pbar.close()
    finally:
        try:
            pbar.close()
        except Exception:
            pass


def _run_pipeline_yolo_gated(
    *,
    cap: cv2.VideoCapture,
    out: cv2.VideoWriter,
    fps: float,
    total_frames: int,
    yolo: YoloDetector,
    lp_detector: LpDetector,
    ocr: Ocr,
    gate: YoloStrideGate,
    trash: "RfDetrTrashDetector",
    times: PipelineStepTimes,
) -> None:
    pbar = tqdm(total=total_frames, desc="Processing video (YOLO stride gate)")
    frame_idx = 0
    try:
        while True:
            t_read = time.perf_counter()
            ret, frame = cap.read()
            times.other_sec += time.perf_counter() - t_read
            if not ret:
                break

            run_yolo = gate.should_run_yolo(frame_idx)
            trash_dets: List[Detection] = []
            if run_yolo:
                fd = FrameData(index=frame_idx, timestamp=frame_idx / fps, image=frame)
                t0 = time.perf_counter()
                dets_list = yolo.detect([fd])
                times.yolo_sec += time.perf_counter() - t0
                detections = dets_list[0] if dets_list else []
                gate.observe(frame_idx, _scene_has_activity(detections, YOLO_CONFIDENCE))
                t0 = time.perf_counter()
                tlist = trash.detect_trash([fd])
                times.trash_sec += time.perf_counter() - t0
                trash_dets = list(tlist[0]) if tlist else []
            else:
                detections = []

            t0 = time.perf_counter()
            _annotate_frame(frame, trash_dets, detections, lp_detector=lp_detector, ocr=ocr)
            times.annotate_sec += time.perf_counter() - t0
            t0 = time.perf_counter()
            out.write(frame)
            times.video_write_sec += time.perf_counter() - t0

            frame_idx += 1
            pbar.update(1)

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

    t0 = time.perf_counter()
    yolo = YoloDetector(conf_threshold=YOLO_CONFIDENCE)
    lp_detector = LpDetector()
    ocr = Ocr()
    trash = _load_trash_detector_required()
    times.init_sec = time.perf_counter() - t0

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

    if mode == "yolo":
        console.print(
            f"[cyan]Loaded video[/] {video_path} ({width}x{height} @ {fps:.2f} FPS, {total_frames} frames); "
            f"GATE_MODE=yolo | coarse_stride={YOLO_COARSE_STRIDE} dense_stride={YOLO_DENSE_STRIDE} "
            f"dense_idle_miss_streak={YOLO_DENSE_IDLE_MISS_STREAK} (YOLO runs w/o person/vehicle to leave dense)"
        )
    else:
        console.print(
            f"[cyan]Loaded video[/] {video_path} ({width}x{height} @ {fps:.2f} FPS, {total_frames} frames); "
            f"chunk={CHUNK_SECONDS}s -> {chunk_frames} frames | GATE_MODE=off"
        )
    console.print("[cyan]RF-DETR trash/cigarette[/] loaded (required)")

    out_dir = os.path.dirname(os.path.abspath(output_video))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
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

    wall_total = time.perf_counter() - wall_start
    times.print_summary(wall_total_sec=wall_total)


if __name__ == "__main__":
    run_pipeline(VIDEO_PATH, OUTPUT_VIDEO)
