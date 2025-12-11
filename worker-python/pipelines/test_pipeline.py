#!/usr/bin/env python3
# worker-python/scripts/test_pipeline.py
"""
Test pipeline: runs YOLO -> LP -> OCR on a local video, writes:
video_name, timestamp, crime, vehicle_number

Usage:
    cd worker-python
    python -m scripts.test_pipeline
"""

import os
import sys
import cv2
from tqdm import tqdm
from rich.console import Console

from models.yolo_detector import YoloDetector
from models.lp_detector import LpDetector
from models.ocr import Ocr  # must return List[Tuple[str, float]] per crop
from core.types import FrameData
from settings import CHUNK_SECONDS, YOLO_CONFIDENCE, PLATE_CONFIDENCE

console = Console()

# Config — edit if needed
VIDEO_PATH = "Test.mp4"
OUTPUT_VIDEO = "output_with_boxes.mp4"

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


def main() -> None:
    if not os.path.exists(VIDEO_PATH):
        console.print(f"[red]Video not found:[/] {VIDEO_PATH}")
        sys.exit(2)

    video_name = os.path.basename(VIDEO_PATH)

    # Initialize models
    yolo = YoloDetector(conf_threshold=YOLO_CONFIDENCE)
    lp_detector = LpDetector()
    ocr = Ocr()  # GPU-based Ocr implementation as requested

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        console.print(f"[red]Failed to open video:[/] {VIDEO_PATH}")
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
        f"[cyan]Loaded video[/] {VIDEO_PATH} ({width}x{height} @ {fps:.2f} FPS, {total_frames} frames);"
        f" chunk={CHUNK_SECONDS}s -> {chunk_frames} frames"
    )

    # Output video writer (annotated visualization)
    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    pbar = tqdm(total=total_frames, desc="Processing video")
    frame_idx = 0

    try:
        while True:
            # Read a chunk of frames
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

            # Run YOLO on the chunk
            detections_per_frame = yolo.detect(chunk_frames_list)

            # Align safety
            n = min(len(chunk_frames_list), len(detections_per_frame))
            if n == 0:
                continue

            for i in range(n):
                frame_data = chunk_frames_list[i]
                frame = frame_data.image
                h, w = frame.shape[:2]
                detections = detections_per_frame[i]

                # Draw YOLO detections
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

                # Handle vehicles
                vehicles = [d for d in detections if d.label in VEHICLE_LABELS and d.confidence >= YOLO_CONFIDENCE]
                for v in vehicles:
                    vb = clamp_bbox(v.bbox, w, h)
                    if vb is None:
                        continue
                    vx1, vy1, vx2, vy2 = vb

                    # crop vehicle and detect plates
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

                        # OCR — expects list of crops, returns list of (text, conf)
                        try:
                            ocr_out = ocr.recognize([plate_crop])
                        except Exception as e:
                            console.print(f"[yellow]OCR error:[/] {str(e)}")
                            ocr_out = [("", 0.0)]

                        plate_text, plate_conf = ocr_out[0] if ocr_out else ("", 0.0)

                        # Optional: filter low confidence
                        if plate_conf < PLATE_CONFIDENCE:
                            continue

                        # draw plate box + text+confidence on frame (convert plate coords to full-frame)
                        cv2.rectangle(frame, (vx1 + px1, vy1 + py1), (vx1 + px2, vy1 + py2), (255, 0, 0), 2)
                        label_str = f"{plate_text} {plate_conf:.2f}" if plate_text else f"{plate_conf:.2f}"
                        cv2.putText(frame, label_str, (vx1 + px1, max(0, vy1 + py1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Write annotated frame
                out.write(frame)

        # finalize
        pbar.close()
        console.print(f"[green]Annotated video saved:[/] {OUTPUT_VIDEO}")

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
    main()