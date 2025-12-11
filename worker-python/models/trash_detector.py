# worker-python/models/trash_detector.py

from pathlib import Path
from typing import List, Optional

from rfdetr import RFDETRLarge
from models.base import TrashDetector
from core.types import FrameData, Detection


class RfDetrTrashDetector(TrashDetector):
    """
    RF-DETR wrapper that runs TWO models:
    1) Trash RF-DETR model (general waste)
    2) Cigarette butt RF-DETR model (single class)

    Outputs are merged into a single list of Detection objects per frame.
    """

    def __init__(
        self,
        weights_path: str | None = None,
        cigarette_weights_path: str | None = None,
        allowed_classes: Optional[List[int]] = None,
        class_names: Optional[dict] = None,
        conf_threshold: float = 0.25,
    ):
        """
        :param weights_path: RF-DETR trash model weights
        :param cigarette_weights_path: RF-DETR cigarette-butt model weights
        :param allowed_classes: optional filtering
        :param class_names: mapping for general trash classes
        :param conf_threshold: confidence threshold
        """

        # ---- TRASH MODEL ----
        if weights_path is None:
            weights_path = (
                Path(__file__).resolve().parents[1] / "weights" / "trash.pth"
            )
        self.trash_model = RFDETRLarge(pretrain_weights=str(weights_path))

        # ---- CIGARETTE MODEL ----
        if cigarette_weights_path is None:
            cigarette_weights_path = (
                Path(__file__).resolve().parents[1] / "weights" / "cigarette.pth"
            )
        self.cig_model = RFDETRLarge(pretrain_weights=str(cigarette_weights_path))

        self.allowed_classes = allowed_classes
        self.class_names = class_names or {}
        self.conf_threshold = conf_threshold

    def _convert_rfdetr_results(self, results, default_label="trash"):
        """
        Helper: Convert RF-DETR model outputs into Detection objects.
        """
        frame_outputs = []

        for det in results:
            xyxy = det.xyxy
            confs = det.confidence
            class_ids = det.class_id

            frame_det_list: List[Detection] = []

            for i in range(len(xyxy)):
                score = float(confs[i])
                cid = int(class_ids[i])

                if score < self.conf_threshold:
                    continue

                if self.allowed_classes and cid not in self.allowed_classes:
                    continue

                x1, y1, x2, y2 = map(float, xyxy[i])

                # Pick label → either from class_names or fallback
                label = self.class_names.get(cid, default_label)

                frame_det_list.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        label=label,
                        confidence=score,
                    )
                )

            frame_outputs.append(frame_det_list)

        return frame_outputs

    def detect_trash(self, frames: List[FrameData]) -> List[List[Detection]]:
        """
        Runs BOTH models and merges detections per frame.
        """
        if not frames:
            return []

        images = [f.image for f in frames]

        # ---- 1) TRASH DETECTION ----
        trash_results = self.trash_model.predict(images)
        trash_dets = self._convert_rfdetr_results(
            trash_results,
            default_label="trash",
        )

        # ---- 2) CIGARETTE DETECTION ----
        cig_results = self.cig_model.predict(images)
        cig_dets = self._convert_rfdetr_results(
            cig_results,
            default_label="cigarette",
        )

        # ---- MERGE BOTH MODELS ----
        merged: List[List[Detection]] = []

        for t_list, c_list in zip(trash_dets, cig_dets):
            merged.append(t_list + c_list)

        return merged
