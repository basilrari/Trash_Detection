"""RF-DETR litter / cigarette heads (``rfdetr`` + local ``.pth`` checkpoints)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

import cv2
import numpy as np
import torch

from core.types import Detection, FrameData
from models.base import TrashDetector


def _ckpt_get(args_obj: Any, key: str) -> Any:
    if args_obj is None:
        return None
    if isinstance(args_obj, Mapping):
        return args_obj.get(key)
    return getattr(args_obj, key, None)


def _rfdetr_kwargs_from_checkpoint(weights_path: str, cfg_class: type[Any]) -> Dict[str, Any]:
    """Pull constructor kwargs from ``checkpoint['args']`` that exist on the model config class."""
    allowed = frozenset(cfg_class.model_fields.keys()) - {"pretrain_weights"}
    p = Path(weights_path)
    if not p.is_file():
        return {}
    try:
        try:
            ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(p), map_location="cpu")
    except Exception:
        return {}
    raw_args = ckpt.get("args")
    out: Dict[str, Any] = {}
    for key in allowed:
        val = _ckpt_get(raw_args, key)
        if val is None:
            continue
        out[key] = val
    return out


def _manual_rfdetr_overrides() -> Dict[str, Any]:
    from settings import (
        RF_DETR_NUM_CLASSES,
        RF_DETR_PATCH_SIZE,
        RF_DETR_POSITIONAL_ENCODING_SIZE,
        RF_DETR_RESOLUTION,
    )

    out: Dict[str, Any] = {}
    if RF_DETR_PATCH_SIZE is not None:
        out["patch_size"] = RF_DETR_PATCH_SIZE
    if RF_DETR_NUM_CLASSES is not None:
        out["num_classes"] = RF_DETR_NUM_CLASSES
    if RF_DETR_RESOLUTION is not None:
        out["resolution"] = RF_DETR_RESOLUTION
    if RF_DETR_POSITIONAL_ENCODING_SIZE is not None:
        out["positional_encoding_size"] = RF_DETR_POSITIONAL_ENCODING_SIZE
    return out


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


def _ensure_positional_encoding_size(merged: Dict[str, Any]) -> None:
    """If ``resolution`` and ``patch_size`` are set but PE is not, set ``PE = resolution // patch_size``."""
    if merged.get("positional_encoding_size") is not None:
        return
    res = merged.get("resolution")
    ps = merged.get("patch_size")
    if res is None or ps is None:
        return
    merged["positional_encoding_size"] = int(res) // int(ps)


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
    cfg_class = ctor._model_config_class

    merged: Dict[str, Any] = dict(_rfdetr_kwargs_from_checkpoint(weights_path, cfg_class))
    merged.update(_manual_rfdetr_overrides())
    _ensure_positional_encoding_size(merged)
    merged["pretrain_weights"] = str(weights_path)
    return ctor(**merged)


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
