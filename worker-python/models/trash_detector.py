"""RF-DETR litter / cigarette heads (``rfdetr`` + local ``.pth`` checkpoints)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np

from core.types import Detection, FrameData
from models.base import TrashDetector


def _sv_to_detections(
    sv_det: Any,
    *,
    class_id_map: Dict[int, str] | None,
    default_label: str,
) -> List[Detection]:
    """Convert ``supervision.Detections`` to our :class:`Detection` list."""
    if sv_det is None or len(sv_det) == 0:
        return []

    xyxy = sv_det.xyxy
    confs = sv_det.confidence
    cls_ids = sv_det.class_id
    names = None
    try:
        names = sv_det.data.get("class_name") if hasattr(sv_det, "data") else None
    except Exception:
        names = None

    out: List[Detection] = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = map(float, xyxy[i])
        cf = float(confs[i]) if confs is not None else 0.0
        cid = int(cls_ids[i]) if cls_ids is not None else 0
        if class_id_map and cid in class_id_map:
            label = class_id_map[cid]
        elif names is not None and i < len(names):
            label = str(names[i]).strip() or default_label
        else:
            label = default_label
        out.append(Detection(bbox=(x1, y1, x2, y2), label=label, confidence=cf))
    return out


def _build_rfdetr(model_size: str, weights_path: str) -> Any:
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    size = (model_size or "medium").strip().lower()
    table = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }
    ctor = table.get(size, RFDETRMedium)
    return ctor(pretrain_weights=str(weights_path))


class RfDetrTrashDetector(TrashDetector):
    """Runs one or two RF-DETR checkpoints (e.g. ``trash.pth`` + ``cigarette.pth``) per batch.

    Expects RGB inference internally; ``detect_trash`` converts BGR ``FrameData`` images.
    """

    def __init__(
        self,
        weights_path: str | Path,
        *,
        cigarette_weights_path: str | Path | None = None,
        class_names: Dict[int, str] | None = None,
        conf_threshold: float = 0.4,
        model_size: str = "medium",
    ) -> None:
        self._conf = float(conf_threshold)
        self._class_names = dict(class_names) if class_names else None
        wp = Path(weights_path)
        if not wp.is_file():
            raise FileNotFoundError(f"Trash RF-DETR weights not found: {wp}")

        self._models: List[tuple[Any, str, str]] = []
        # (rfdetr_model, default_label_for_cls_fallback, run_tag)
        self._models.append((_build_rfdetr(model_size, str(wp)), "trash", "trash"))

        if cigarette_weights_path:
            cp = Path(cigarette_weights_path)
            if cp.is_file() and cp.resolve() != wp.resolve():
                self._models.append(
                    (_build_rfdetr(model_size, str(cp)), "cigarette", "cigarette")
                )

    def detect_trash(self, frames: List[FrameData]) -> List[List[Detection]]:
        if not frames:
            return []
        images_rgb = [cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB) for f in frames]
        merged: List[List[Detection]] = [[] for _ in frames]

        for model, default_lbl, _tag in self._models:
            raw = model.predict(images_rgb, threshold=self._conf)
            per_frame = raw if isinstance(raw, list) else [raw]
            if len(per_frame) != len(frames):
                continue
            for i, sv_det in enumerate(per_frame):
                merged[i].extend(
                    _sv_to_detections(
                        sv_det,
                        class_id_map=self._class_names,
                        default_label=default_lbl,
                    )
                )
        return merged
