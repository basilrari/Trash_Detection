from pathlib import Path
from typing import List

from ultralytics import YOLO

from models.base import LicensePlateDetector
from core.types import FrameData, LicensePlate


class LpDetector(LicensePlateDetector):
    def __init__(self, weights_path: str | None = None, plate_class_id: int = 0):
        """
        License plate detector based on a YOLO model.

        :param weights_path: Optional custom path to LP model weights.
        :param plate_class_id: Class index corresponding to plates in the model.
        """
        if weights_path is None:
            weights_path = (
                Path(__file__).resolve().parents[1] / "weights" / "bestlicense.pt"
            )

        self.model = YOLO(str(weights_path))
        self.plate_class_id = plate_class_id

    def detect_plates(self, frames: List[FrameData]) -> List[List[LicensePlate]]:
        """
        Run license plate detection on a batch of frames (usually vehicle crops).

        :param frames: List of FrameData with .image as cropped vehicle regions.
        :return: List (per frame) of LicensePlate objects.
        """
        if not frames:
            return []

        images = [f.image for f in frames]

        results = self.model(
            images,
            classes=[self.plate_class_id],
        )

        all_plates: List[List[LicensePlate]] = []

        for result in results:
            frame_plates: List[LicensePlate] = []

            for box in result.boxes:
                conf = float(box.conf)
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                bbox = (x1, y1, x2, y2)

                # text will be filled in by OCR later
                lp = LicensePlate(bbox=bbox, text="", confidence=conf)
                frame_plates.append(lp)

            all_plates.append(frame_plates)

        return all_plates
