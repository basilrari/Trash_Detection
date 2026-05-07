#!/usr/bin/env python3
"""
YOLO → LP detector → OCR on a local video; writes an annotated MP4.

Run from worker-python/:

  python worker.py
  python worker.py myvideo.mp4 -o out.mp4
  python -m pipelines.test_pipeline   # uses paths from settings.py
"""

import os
import sys
import cv2
from tqdm import tqdm
from rich.console import Console

from models.yolo_detector import YoloDetector
from models.lp_detector import LpDetector
from models.ocr import Ocr
from core.types import FrameData
from settings import CHUNK_SECONDS, OUTPUT_VIDEO, PLATE_CONFIDENCE, VIDEO_PATH, YOLO_CONFIDENCE

console = Console()

VEHICLE_LABELS = ("vehicle", "car", "truck", "bus", "motorbike", "motorcycle")


def clamp_bbox(bbox, w, h):
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def run_pipeline(video_path: str, output_video: str) -> None:
    """Process ``video_path`` and write annotated video to ``output_video``."""
    if not os.path.exists(video_path):
        console.print(f"[red]Video not found:[/] {video_path}")
        sys.exit(2)

    yolo = YoloDetector(conf_threshold=YOLO_CONFIDENCE)
    lp_detector = LpDetector()
    ocr = Ocr()

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
        f"[cyan]Loaded video[/] {video_path} ({width}x{height} @ {fps:.2f} FPS, {total_frames} frames);"
        f" chunk={CHUNK_SECONDS}s -> {chunk_frames} frames"
    )

    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    pbar = tqdm(total=total_frames, desc="Processing video")
    frame_idx = 0

    try:
        while True:
            chunk_frames_list = []
            for _ in range(chunk_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                timestamp = frame_idx / fps
                chunk_frames_list.append(FrameData(index=frame_idx, timestamp=timestamp, image=frame))
                frame_idx += 1
                pbar.update(1)

            if not chunk_frames_list:
                break

            detections_per_frame = yolo.detect(chunk_frames_list)

            n = min(len(chunk_frames_list), len(detections_per_frame))
            if n == 0:
                continue

            for i in range(n):
                frame_data = chunk_frames_list[i]
                frame = frame_data.image
                h, w = frame.shape[:2]
                detections = detections_per_frame[i]

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
                    cv2.putText(frame, f"{det.label} {det.confidence:.2f}", (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
                        cv2.putText(frame, label_str, (vx1 + px1, max(0, vy1 + py1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                out.write(frame)

        pbar.close()
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


if __name__ == "__main__":
    run_pipeline(VIDEO_PATH, OUTPUT_VIDEO)
