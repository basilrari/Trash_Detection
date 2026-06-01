"""
External detector → peeing/pose contract (DFINE, RT-DETR, etc.).

**This kit does not run a scene detector.** Your repo (e.g. DFINE) must supply person boxes
in **full-frame pixel coordinates** on the same BGR frame you pass to ``PeeingDetector.update``.

Quick path
----------
1. Run DFINE (or your detector) on each **stride-sampled** frame (see ``pipelines.frame_stride``).
2. Convert outputs to ``list[Detection]`` (helpers below).
3. Pass that list into ``PeeingDetector.update(frame_bgr, detections, run_yolo=True, ...)``.
4. On skipped frames, pass the **last** detection list and ``run_yolo=False`` (tracks stay alive).

**Required classes (every sampled frame):**

- ``person`` — pose / peeing heuristic runs on these boxes.
- ``motorcycle`` and/or ``motorbike`` — **must** be produced by your detector and included in the
  same list (use ``merge_person_and_motorcycle_detections``). The seated-rider gate needs them;
  person-only lists are not supported for production integration.

``Detection`` shape (``models.types``)::

    Detection(
        bbox=(x1, y1, x2, y2),   # float pixels, origin top-left, x2>x1, y2>y1
        label="person",          # or ``"motorcycle"`` / ``"motorbike"`` for the gate
        confidence=0.92,         # compared to ``min_detection_conf`` in update()
    )
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Union

from models.types import Detection

# Labels consumed by PeeingDetector (must match ``peeing_detector.PERSON_LABELS`` / gate labels).
PERSON_LABEL = "person"
MOTORCYCLE_LABELS = frozenset({"motorcycle", "motorbike"})


def detection_from_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    label: str = PERSON_LABEL,
    confidence: float = 1.0,
) -> Detection:
    """Build one ``Detection`` from a pixel axis-aligned box."""
    return Detection(
        bbox=(float(x1), float(y1), float(x2), float(y2)),
        label=str(label),
        confidence=float(confidence),
    )


def person_detections_from_arrays(
    boxes_xyxy: Sequence[Sequence[float]],
    scores: Sequence[float] | None = None,
    *,
    default_score: float = 1.0,
) -> List[Detection]:
    """
    Convert a batch of person boxes (N×4) to ``Detection`` rows.

    Typical DFINE / torchvision output after converting to xyxy in **original image space**::

        dets = person_detections_from_arrays(boxes.cpu().numpy(), scores.cpu().numpy())
    """
    out: List[Detection] = []
    for i, box in enumerate(boxes_xyxy):
        if len(box) != 4:
            raise ValueError(f"box[{i}] must have 4 elements (x1,y1,x2,y2), got {box!r}")
        conf = float(scores[i]) if scores is not None else float(default_score)
        out.append(detection_from_xyxy(box[0], box[1], box[2], box[3], confidence=conf))
    return out


def prepare_scene_detections(
    detections: Iterable[Detection],
    *,
    min_confidence: float = 0.0,
    keep_motorcycles: bool = True,
) -> List[Detection]:
    """
    Normalize a mixed detector output list for ``PeeingDetector.update``.

    - Keeps ``person`` with ``confidence >= min_confidence``.
    - Keeps ``motorcycle`` / ``motorbike`` with ``confidence >= min_confidence`` (required labels
      when your detector emits them; call ``merge_person_and_motorcycle_detections`` first).
    - Drops other classes (cars, etc.).
    """
    out: List[Detection] = []
    for d in detections:
        if d.label == PERSON_LABEL and d.confidence >= min_confidence:
            out.append(d)
        elif keep_motorcycles and d.label in MOTORCYCLE_LABELS and d.confidence >= min_confidence:
            out.append(d)
    return out


def merge_person_and_motorcycle_detections(
    person_boxes: Sequence[Sequence[float]],
    person_scores: Sequence[float],
    motorcycle_boxes: Sequence[Sequence[float]],
    motorcycle_scores: Sequence[float],
    *,
    motorcycle_label: str = "motorcycle",
) -> List[Detection]:
    """
    Build the combined list expected by ``PeeingDetector.update``.

    **You must call this (or equivalent)** so every frame list includes both person and
    motorcycle/motorbike rows. Pass empty ``motorcycle_boxes`` only when the detector found no
    bikes in that frame — not because motorcycle inference was skipped.
    """
    out = person_detections_from_arrays(person_boxes, person_scores)
    ms = motorcycle_scores
    for box, sc in zip(motorcycle_boxes, ms):
        out.append(
            detection_from_xyxy(
                box[0], box[1], box[2], box[3],
                label=motorcycle_label,
                confidence=float(sc),
            )
        )
    return out
