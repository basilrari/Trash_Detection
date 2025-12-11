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
import supervision as sv
from models.yolo_detector import YoloDetector
from models.lp_detector import LpDetector
from models.ocr import Ocr  # must return List[Tuple[str, float]] per crop
from core.types import FrameData
from models.trash_detector import RfDetrTrashDetector
from settings import CHUNK_SECONDS, YOLO_CONFIDENCE, PLATE_CONFIDENCE
import numpy as np

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
    trash_detector = RfDetrTrashDetector(
    weights_path="weights/trash.pth",   # <-- update with your file
    class_names={1: "trash"},  # optional
    conf_threshold=0.40
    )

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
            # Run RF-DETR trash detection on the same chunk
            trash_detections_per_frame = trash_detector.detect_trash(chunk_frames_list)

            # Align safety
            n = min(len(chunk_frames_list), len(detections_per_frame))
            if n == 0:
                continue

            for i in range(n):
                frame_data = chunk_frames_list[i]
                frame = frame_data.image
                h, w = frame.shape[:2]
            
                ##########################################
                # SUPER VISION VISUALIZATION SETUP
                ##########################################
                text_scale = sv.calculate_optimal_text_scale(resolution_wh=(w, h))
                thickness = sv.calculate_optimal_line_thickness(resolution_wh=(w, h))
            
                trash_box_annotator = sv.BoxAnnotator(color=sv.Color.RED, thickness=thickness)
                yolo_box_annotator  = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=thickness)
                plate_box_annotator = sv.BoxAnnotator(color=sv.Color.BLUE, thickness=thickness)

            
                label_annotator = sv.LabelAnnotator(
                    text_color=sv.Color.BLACK,
                    text_scale=text_scale,
                    text_thickness=thickness,
                    smart_position=True
                )
            
                # Supervision expects a detections object
                # We will build *three* supervision detections:
                # 1. trash_dets_s (RED)
                # 2. yolo_dets_s  (GREEN)
                # 3. plate_dets_s (BLUE)
            
                ##########################################
                # TRASH DETECTIONS (RED)
                ##########################################
                trash_detections = trash_detections_per_frame[i]

                if len(trash_detections) > 0:
                    trash_xyxy = []
                    trash_conf = []
                    trash_class = []
                
                    for t in trash_detections:
                        x1, y1, x2, y2 = map(int, t.bbox)
                        trash_xyxy.append([x1, y1, x2, y2])
                        trash_conf.append(t.confidence)
                        trash_class.append(0)
                
                    trash_dets_s = sv.Detections(
                        xyxy=np.array(trash_xyxy),
                        confidence=np.array(trash_conf),
                        class_id=np.array(trash_class)
                    )
                
                    trash_labels = [f"{t.label} {t.confidence:.2f}" for t in trash_detections]
                
                    frame = trash_box_annotator.annotate(frame, trash_dets_s)

                    frame = label_annotator.annotate(
                        scene=frame,
                        detections=trash_dets_s,
                        labels=trash_labels
                    )
            
                ##########################################
                # YOLO DETECTIONS (GREEN)
                ##########################################
                ##########################################

                detections = detections_per_frame[i]
                
                if len(detections) > 0:
                    yolo_xyxy, yolo_conf, yolo_class, yolo_labels = [], [], [], []
                
                    for d in detections:
                        x1, y1, x2, y2 = map(int, d.bbox)
                        yolo_xyxy.append([x1, y1, x2, y2])
                        yolo_conf.append(d.confidence)
                        yolo_class.append(0)
                        yolo_labels.append(f"{d.label} {d.confidence:.2f}")
                
                    yolo_dets_s = sv.Detections(
                        xyxy=np.array(yolo_xyxy),
                        confidence=np.array(yolo_conf),
                        class_id=np.array(yolo_class)
                    )
                
                    frame = yolo_box_annotator.annotate(frame, yolo_dets_s)

                    frame = label_annotator.annotate(frame, yolo_dets_s, yolo_labels)

            
                ##########################################
                # LICENSE PLATE DETECTIONS (BLUE)
                ##########################################
                plate_xyxy = []
                plate_conf = []
                plate_class = []
                plate_labels = []
                
                vehicles = [d for d in detections if d.label in VEHICLE_LABELS]
                
                for v in vehicles:
                    vx1, vy1, vx2, vy2 = map(int, v.bbox)
                
                    # Crop the vehicle area
                    vehicle_crop = frame[vy1:vy2, vx1:vx2]
                    if vehicle_crop.size == 0:
                        continue
                
                    plates_list = lp_detector.detect_plates(
                        [FrameData(0, 0.0, vehicle_crop)]
                    )
                
                    if not plates_list:
                        continue
                
                    plates = plates_list[0]
                
                    for plate in plates:
                        px1, py1, px2, py2 = map(int, plate.bbox)
                        full_x1 = vx1 + px1
                        full_y1 = vy1 + py1
                        full_x2 = vx1 + px2
                        full_y2 = vy1 + py2
                
                        plate_xyxy.append([full_x1, full_y1, full_x2, full_y2])
                        plate_conf.append(plate.confidence)
                        plate_class.append(0)
                        plate_labels.append(f"plate {plate.confidence:.2f}")
                
                # Only annotate if we detected plates
                if len(plate_xyxy) > 0:
                    plate_dets_s = sv.Detections(
                        xyxy=np.array(plate_xyxy),
                        confidence=np.array(plate_conf),
                        class_id=np.array(plate_class)
                    )
                
                    frame = plate_box_annotator.annotate(frame, plate_dets_s)
                    frame = label_annotator.annotate(frame, plate_dets_s, plate_labels)

                ##########################################
                # WRITE FRAME
                ##########################################
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