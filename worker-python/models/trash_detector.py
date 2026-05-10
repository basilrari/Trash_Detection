"""Shared helpers for RF-DETR TensorRT trash + cigarette heads (see ``rfdetr_trt_trash``)."""

from __future__ import annotations

from typing import Any, Dict, List

from models.types import Detection

# Supervision / RF-DETR sometimes attach placeholder ``class_name`` strings instead of
# dataset labels; never show these when we still have a meaningful head default.
_JUNK_SV_CLASS_NAMES = frozenset(
    {
        "",
        "0",
        "_background",
        "background",
        "__background__",
        "bg",
        "n/a",
        "na",
    }
)


def _label_from_sv_class_name(raw: Any, default_label: str) -> str:
    s = str(raw).strip()
    if not s or s.lower() in _JUNK_SV_CLASS_NAMES:
        return default_label
    return s


def _sv_to_detections(
    sv_det: Any,
    *,
    class_id_map: Dict[int, str] | None,
    default_label: str,
    use_sv_class_names: bool = True,
) -> List[Detection]:
    """Convert ``supervision.Detections`` to our :class:`Detection` list.

    When ``use_sv_class_names`` is False (default for dedicated RF-DETR heads),
    RF-DETR / supervision string placeholders like ``\"0\"`` or ``\"_background\"``
    are ignored and every box uses ``default_label`` (unless ``class_id_map`` hits).
    """
    if sv_det is None or len(sv_det) == 0:
        return []

    xyxy = sv_det.xyxy
    confs = sv_det.confidence
    cls_ids = sv_det.class_id
    names = None
    if use_sv_class_names:
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
        elif use_sv_class_names and names is not None and i < len(names):
            label = _label_from_sv_class_name(names[i], default_label)
        else:
            label = default_label
        out.append(Detection(bbox=(x1, y1, x2, y2), label=label, confidence=cf))
    return out
