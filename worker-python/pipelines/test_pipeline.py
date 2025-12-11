# worker-python/scripts/test_pipeline.py

import cv2
import numpy as np
from tqdm import tqdm
from rich.console import Console

from models.yolo_detector import YoloDetector
from models.lp_detector import LpDetector
from models.ocr import Ocr
from core.types import FrameData
from settings import CHUNK_SECONDS, YOLO_CONFIDENCE  # adjust path if needed

console = Console()

VIDEO_PATH = "Test.mp4"

def main() -> None:
    # Initialize models (adjust constructor args to your actual signatures)
    yolo = YoloDetector(conf_threshold=YOLO_CONFIDENCE)
    lp_detector = LpDetector()
    ocr = Ocr()

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS ({fps}) for video: {VIDEO_PATH}")

    chunk_size = max(1, int(CHUNK_SECONDS * fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    console.print(
        f"[cyan]Loaded video[/] {VIDEO_PATH} "
        f"({width}x{height} @ {fps:.2f} FPS, {total_frames} frames, "
        f"chunk_size={chunk_size})"
    )

    # Output video writer
    out = cv2.VideoWriter(
        "output_with_boxes.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    progress = tqdm(total=total_frames, desc="Processing video")

    frame_idx = 0
    while cap.isOpened():
        chunk_frames: list[FrameData] = []

        # Read a chunk of frames
        for _ in range(chunk_size):
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            chunk_frames.append(FrameData(index=frame_idx, timestamp=timestamp, image=frame))
            frame_idx += 1
            progress.update(1)

        if not chunk_frames:
            break

        # Run YOLO (people + vehicles)
        # Assumes yolo.detect returns list[list[Detection]] aligned with chunk_frames
        detections_per_frame = yolo.detect(chunk_frames)

        # Safety check: lengths should match
        if len(detections_per_frame) != len(chunk_frames):
            console.print(
                f"[yellow]Warning:[/] detections_per_frame length "
                f"({len(detections_per_frame)}) != chunk_frames length "
                f"({len(chunk_frames)}). Check YoloDetector implementation."
            )

        for idx, frame_detections in enumerate(detections_per_frame):
            frame_data = chunk_frames[idx]
            frame = frame_data.image
            h, w = frame.shape[:2]

            for det in frame_detections:
                x1, y1, x2, y2 = map(int, det.bbox)

                # Clamp to frame bounds
                x1 = max(0, min(w, x1))
                x2 = max(0, min(w, x2))
                y1 = max(0, min(h, y1))
                y2 = max(0, min(h, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                # Draw YOLO bbox
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

                # If this detection is a vehicle, run LP + OCR
                # Adjust labels per your model (e.g., "car", "truck", etc.)
                if det.label in ("vehicle", "car", "truck", "bus", "motorbike", "motorcycle"):
                    vehicle_crop = frame[y1:y2, x1:x2]
                    if vehicle_crop.size == 0:
                        continue

                    # LP detector expects list[FrameData], here we wrap one crop
                    lp_frame = FrameData(index=0, timestamp=0.0, image=vehicle_crop)
                    plate_detections_per_frame = lp_detector.detect_plates([lp_frame])
                    if not plate_detections_per_frame:
                        continue

                    plate_detections = plate_detections_per_frame[0]

                    for plate in plate_detections:
                        px1, py1, px2, py2 = map(int, plate.bbox)

                        # Clamp to crop bounds
                        ph, pw = vehicle_crop.shape[:2]
                        px1 = max(0, min(pw, px1))
                        px2 = max(0, min(pw, px2))
                        py1 = max(0, min(ph, py1))
                        py2 = max(0, min(ph, py2))
                        if px2 <= px1 or py2 <= py1:
                            continue

                        plate_crop = vehicle_crop[py1:py2, px1:px2]
                        if plate_crop.size == 0:
                            continue

                        # OCR expects list[np.ndarray]
                        ocr_results = ocr.recognize([plate_crop])
                        # Adjust based on your Ocr API (string vs. (text, conf) etc.)
                        plate_text = ocr_results[0] if ocr_results else ""

                        # Draw plate box and text back in the original frame coordinates
                        cv2.rectangle(
                            frame,
                            (x1 + px1, y1 + py1),
                            (x1 + px2, y1 + py2),
                            (255, 0, 0),
                            2,
                        )
                        if plate_text:
                            cv2.putText(
                                frame,
                                plate_text,
                                (x1 + px1, max(0, y1 + py1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 0),
                                2,
                            )

            # Write annotated frame
            out.write(frame)

    cap.release()
    out.release()
    progress.close()
    console.print("[green]Test complete — output video saved as 'output_with_boxes.mp4'[/]")


if __name__ == "__main__":
    main()
