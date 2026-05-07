#!/usr/bin/env python3
"""
YOLO → RF-DETR trash → LP → OCR on a local video; writes an annotated MP4.
Labels only (no bounding boxes) for trash, YOLO, and plates; YOLO is person + road vehicles.

Run from worker-python/ (put source videos in inputs/, results under outputs/ by default):

  python worker.py
  python worker.py inputs/myvideo.mp4
  python worker.py inputs/myvideo.mp4 -o outputs/custom.mp4
  python -m pipelines.test_pipeline   # uses paths from settings.py

**Gate (``GATE_MODE``)** — default ``yolo``; see ``settings.py`` and ``Readme.md`` § Gating.

**RF-DETR** — required: ``pip install rfdetr``, **both** ``weights/trash.pth`` and ``weights/cigarette.pth``.
See ``models/trash_detector.py``.
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Sequence

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from rich.console import Console

from models.yolo_detector import YoloDetector
from models.lp_detector import LpDetector
from models.ocr import Ocr
from models.peeing_detector import PeeingDetector, PeeingState
from core.types import Detection, FrameData
from core.yolo_stride_gate import YoloStrideGate, YoloStrideGateConfig
from settings import (
    CHUNK_SECONDS,
    CIGARETTE_WEIGHTS_PATH,
    GATE_MODE,
    OUTPUT_VIDEO,
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
    peeing_sec: float = 0.0
    annotate_sec: float = 0.0
    video_write_sec: float = 0.0
    other_sec: float = 0.0  # open video, build writer, gate setup, chunk assembly

    def print_summary(self, *, wall_total_sec: float) -> None:
        inf = self.yolo_sec + self.trash_sec + self.peeing_sec + self.annotate_sec + self.video_write_sec
        console.print("[bold]Step timings (cumulative)[/]")
        console.print(f"  Model init:           {self.init_sec:8.2f} s")
        console.print(f"  Other (I/O, chunks):  {self.other_sec:8.2f} s")
        console.print(f"  YOLO:                 {self.yolo_sec:8.2f} s")
        console.print(f"  RF-DETR (trash):      {self.trash_sec:8.2f} s")
        console.print(f"  Peeing (MediaPipe):   {self.peeing_sec:8.2f} s")
        console.print(f"  Annotate + LP + OCR: {self.annotate_sec:8.2f} s")
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


def _load_trash_detector_required() -> "RfDetrTrashDetector":
    """Load both RF-DETR heads; exit if ``rfdetr`` or either weights file is missing."""
    tw = Path(TRASH_WEIGHTS_PATH)
    cig = Path(CIGARETTE_WEIGHTS_PATH)
    if not tw.is_file():
        console.print(
            f"[red]RF-DETR is required but trash weights not found:[/] {tw}\n"
            "Place your checkpoint at worker-python/weights/trash.pth."
        )
        raise SystemExit(2)
    if not cig.is_file():
        console.print(
            f"[red]RF-DETR is required but cigarette weights not found:[/] {cig}\n"
            "Place your checkpoint at worker-python/weights/cigarette.pth."
        )
        raise SystemExit(2)
    if tw.resolve() == cig.resolve():
        console.print("[red]trash.pth and cigarette.pth must be two different files.[/]")
        raise SystemExit(2)
    from models.trash_detector import RfDetrTrashDetector

    try:
        return RfDetrTrashDetector(
            tw,
            cig,
            class_names=None,
            conf_threshold=TRASH_CONFIDENCE,
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
) -> None:
    """Draw YOLO labels (no boxes), then LP + OCR on vehicle crops (mutates ``frame`` in place)."""
    h, w = frame.shape[:2]
    scene = _filter_scene_detections(detections)

    yolo_dets, yolo_labels = _detections_to_sv(scene, w, h)
    if yolo_labels:
        annots.yolo_label.annotate(frame, yolo_dets, labels=yolo_labels)

    vehicles = [d for d in scene if d.label in VEHICLE_LABELS and d.confidence >= YOLO_CONFIDENCE]
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

        plate_rows: list[list[float]] = []
        plate_confs: list[float] = []
        plate_label_strs: list[str] = []
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
            plate_text = normalize_plate_text(plate_text)

            if plate_conf < PLATE_CONFIDENCE:
                continue

            gx1, gy1, gx2, gy2 = vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2
            gbox = clamp_bbox((gx1, gy1, gx2, gy2), w, h)
            if gbox is None:
                continue
            gx1, gy1, gx2, gy2 = gbox
            plate_rows.append([float(gx1), float(gy1), float(gx2), float(gy2)])
            plate_confs.append(float(plate_conf))
            label_str = f"{plate_text} {plate_conf:.2f}" if plate_text else f"{plate_conf:.2f}"
            plate_label_strs.append(label_str)

        if plate_rows:
            p_xyxy = np.asarray(plate_rows, dtype=np.float32)
            p_conf = np.asarray(plate_confs, dtype=np.float32)
            p_cls = np.zeros(len(plate_rows), dtype=np.int64)
            p_dets = sv.Detections(xyxy=p_xyxy, confidence=p_conf, class_id=p_cls)
            annots.plate_label.annotate(frame, p_dets, labels=plate_label_strs)


def _draw_peeing_overlay(frame: np.ndarray, state: PeeingState) -> None:
    """Large top-left banner: algorithmic CONFIRMED / SUSPECTED / UNSURE (not human verification)."""
    h, _w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = float(max(1.15, min(3.4, h / 380.0)))
    thick = max(2, int(round(scale * 2.0)))
    line_gap = int(6 + h / 160)

    tier = state.status
    line2 = {"confirmed": "CONFIRMED", "suspected": "SUSPECTED", "unsure": "UNSURE"}.get(
        tier, "UNSURE"
    )
    line1 = "PEEING"
    colors = {
        "confirmed": ((50, 255, 255), (0, 0, 0)),
        "suspected": ((60, 180, 255), (0, 0, 0)),
        "unsure": ((190, 190, 190), (20, 20, 20)),
    }
    fill, outline = colors.get(tier, colors["unsure"])

    def line_size(text: str) -> tuple[int, int, int]:
        (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
        return tw, th, bl

    w1, h1, b1 = line_size(line1)
    w2, h2, b2 = line_size(line2)
    tw = max(w1, w2)

    sub_scale = max(0.42, scale * 0.38)
    sub_th = max(1, thick - 1)
    sub_text = f"window hits {state.score:.0%}  (auto)"
    (sw, sh), sbl = cv2.getTextSize(sub_text, font, sub_scale, sub_th)

    pad_x, pad_y = 18, 16
    ox = 14
    top = 16
    baseline1 = top + pad_y + h1
    baseline2 = baseline1 + b1 + line_gap + h2
    baseline3 = baseline2 + b2 + max(8, int(h / 90)) + sh

    box_top = top
    box_bottom = int(baseline3 + sbl + pad_y)
    left = ox - pad_x
    right = ox + max(tw, sw) + pad_x

    overlay = frame.copy()
    cv2.rectangle(overlay, (left, box_top), (right, box_bottom), (24, 24, 24), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (left, box_top), (right, box_bottom), (90, 90, 90), 2)

    def put_outline(text: str, x: int, y_baseline: int) -> None:
        for dx, dy in (
            (-2, 0),
            (2, 0),
            (0, -2),
            (0, 2),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        ):
            cv2.putText(
                frame,
                text,
                (x + dx, y_baseline + dy),
                font,
                scale,
                outline,
                thick + 2,
                cv2.LINE_AA,
            )
        cv2.putText(frame, text, (x, y_baseline), font, scale, fill, thick, cv2.LINE_AA)

    put_outline(line1, ox, baseline1)
    put_outline(line2, ox, baseline2)
    cv2.putText(
        frame,
        sub_text,
        (ox, baseline3),
        font,
        sub_scale,
        (140, 140, 140),
        sub_th,
        cv2.LINE_AA,
    )


def _annotate_frame(
    frame,
    trash_detections: Sequence[Detection],
    yolo_detections: Sequence[Detection],
    *,
    lp_detector: LpDetector,
    ocr: Ocr,
    annots: FrameAnnotators,
    peeing_state: PeeingState,
) -> None:
    _draw_trash_detections(frame, trash_detections, annots)
    _annotate_yolo_lp_ocr(frame, yolo_detections, lp_detector=lp_detector, ocr=ocr, annots=annots)
    _draw_peeing_overlay(frame, peeing_state)


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
            t0 = time.perf_counter()
            trash_per_frame = trash.detect_trash(chunk_frames_list)
            times.trash_sec += time.perf_counter() - t0

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
                t0 = time.perf_counter()
                _annotate_frame(
                    frame,
                    trash_per_frame[i],
                    scene_dets,
                    lp_detector=lp_detector,
                    ocr=ocr,
                    annots=annots,
                    peeing_state=pstate,
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
    annots: FrameAnnotators,
    peeing: PeeingDetector,
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
            scene_dets: List[Detection] = []
            if run_yolo:
                fd = FrameData(index=frame_idx, timestamp=frame_idx / fps, image=frame)
                t0 = time.perf_counter()
                dets_list = yolo.detect([fd])
                times.yolo_sec += time.perf_counter() - t0
                detections = dets_list[0] if dets_list else []
                scene_dets = _filter_scene_detections(detections)
                gate.observe(frame_idx, _scene_has_activity(scene_dets, YOLO_CONFIDENCE))
                t0 = time.perf_counter()
                tlist = trash.detect_trash([fd])
                times.trash_sec += time.perf_counter() - t0
                trash_dets = list(tlist[0]) if tlist else []

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

            t0 = time.perf_counter()
            _annotate_frame(
                frame,
                trash_dets,
                scene_dets,
                lp_detector=lp_detector,
                ocr=ocr,
                annots=annots,
                peeing_state=pstate,
            )
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
    console.print(
        "[cyan]Peeing hint:[/] standing + squat cues; straddle penalty; "
        f"alarm: last {PEEING_WINDOW_SEC:.0f}s of pose hits (score ≥{PEEING_POSE_MATCH_THRESHOLD:.0%}); "
        f"arm when >{PEEING_ALARM_ENTER_HIT_FRACTION:.0%} hits with ≥{PEEING_ALARM_MIN_SAMPLES} samples, "
        f"disarm when <{PEEING_ALARM_EXIT_HIT_FRACTION:.0%} (no per-person IDs)."
    )

    annots = _make_frame_annotators(width, height)

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
