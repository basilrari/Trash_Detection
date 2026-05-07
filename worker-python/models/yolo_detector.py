from pathlib import Path
from typing import List

from ultralytics import YOLO

from models.base import Detector
from core.types import FrameData, Detection


class YoloDetector(Detector):
    def __init__(self, weights_path: str | None = None, conf_threshold: float = 0.5):
        """
        YOLO detector for persons + vehicles.

        :param weights_path: Optional custom path to YOLO weights.
        :param conf_threshold: Confidence threshold for detections.
        """
        if weights_path is None:
            # Resolve ../weights/yolo11x.pt relative to this file
            weights_path = (
                Path(__file__).resolve().parents[1] / "weights" / "yolo11x.pt"
            )

        self.model = YOLO(str(weights_path))
        self.conf_threshold = conf_threshold

        # COCO subset: person + road vehicles only (no train/boat/bicycle/etc.).
        # 0 person, 2 car, 3 motorcycle, 5 bus, 7 truck
        self.classes = [0, 2, 3, 5, 7]

    def detect(self, frames: List[FrameData]) -> List[List[Detection]]:
        """
        Run YOLO on a batch of frames.

        :param frames: List of FrameData with .image as a BGR ndarray.
        :return: List (per frame) of Detection objects.
        """
        if not frames:
            return []

        images = [f.image for f in frames]

        results = self.model(
            images,
            classes=self.classes,
            conf=self.conf_threshold,
        )

        detections_per_frame: List[List[Detection]] = []

        for result in results:
            frame_dets: List[Detection] = []
            names = result.names  # id -> label

            for box in result.boxes:
                cls_id = int(box.cls)
                label = names.get(cls_id, str(cls_id))
                conf = float(box.conf)
                # xyxy is shape (1, 4) or (4,), convert to tuple[float, float, float, float]
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                bbox = (x1, y1, x2, y2)

                frame_dets.append(Detection(bbox=bbox, label=label, confidence=conf))

            detections_per_frame.append(frame_dets)

        return detections_per_frame
