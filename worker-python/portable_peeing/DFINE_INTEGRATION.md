# Integrating DFINE (or any detector) with portable peeing

Trash_Detection used **scene YOLO** for person boxes. **This kit does not.** You run **DFINE** (or your detector) in the host repo and pass boxes into **`PeeingDetector`**.

## Data flow

```text
  Video frame (BGR, full resolution)
        │
        ▼
  [Your DFINE]  ──►  list[Detection]  persons + motorcycles/motorbikes (required classes)
        │
        ▼
  PeeingDetector.update(frame, detections, run_yolo=True, ...)
        │
        ▼
  [YOLO pose TRT] on each gated person crop  →  standing / groin / stillness / 10s rule
        │
        ▼
  PeeingState (active, mark_bboxes, edge_enter, …)
```

## Required: persons **and** motorcycles

Every call to ``PeeingDetector.update`` (via your ``scene_detector``) must pass a **single** ``list[Detection]`` that includes:

1. All **`person`** boxes from DFINE (pose runs on these).
2. All **`motorcycle`** / **`motorbike`** boxes from DFINE (seated-rider gate — **not optional**).

Use ``merge_person_and_motorcycle_detections`` in ``models/detection_contract.py``. If a frame has no bikes, pass empty motorcycle arrays — but your detector must still be configured to output those classes.

## What you must pass

| Field | Requirement |
|--------|----------------|
| **Image** | Same `numpy` BGR frame DFINE was run on (`H×W×3`, `uint8`). |
| **bbox** | `(x1, y1, x2, y2)` **pixel** coords in that frame, origin top-left. |
| **label** | `"person"` for every box that should get pose. |
| **confidence** | Float; compared to `PEEING_DETECTION_CONFIDENCE` in `settings.py` (default `0.4`). |
| **Motorcycles (required)** | You **must** run DFINE (or your detector) for **`motorcycle` and `motorbike`** and pass those boxes in the **same** `list[Detection]` as persons on every sampled frame. The seated-rider gate drops false peeing on riders; without bike boxes the gate cannot work. If the frame has no bikes, pass no motorcycle rows — but do not skip the motorcycle classes in your detector setup. Labels: exactly `"motorcycle"` or `"motorbike"` (see `PEEING_MOTORCYCLE_LABELS` in `settings.py`). |

Helpers: `models/detection_contract.py` (`detection_from_xyxy`, `person_detections_from_arrays`, `prepare_scene_detections`).

## Stride (when to call DFINE)

Peeing pose runs on **sampled** frames only (`frame_index % stride == 0`). Stride comes from `pipelines.frame_stride._resolve_frame_sample_stride(fps)` (default ~4 scene frames per second of video).

- **On a sampled frame:** run DFINE → pass detections with `run_yolo=True`.
- **On other frames:** pass the **previous** detection list unchanged with `run_yolo=False` (IoU tracks and stillness still advance from bbox positions on sampled frames only).

The video helper `run_peeing_pipeline_video(..., scene_detector=your_fn)` implements this carry for you.

## Minimal loop (single frame)

```python
import cv2
from models.detection_contract import (
    merge_person_and_motorcycle_detections,
    prepare_scene_detections,
)
from models.peeing_detector import PeeingDetector
from settings import PEEING_DETECTION_CONFIDENCE

peeing = PeeingDetector()  # loads pose TRT from settings
frame = cv2.imread("frame.jpg")

# --- your DFINE forward; boxes must be in full-frame pixels ---
p_boxes, p_scores = run_dfine_person(frame)
m_boxes, m_scores = run_dfine_motorcycles(frame)  # required class(es)

detections = prepare_scene_detections(
    merge_person_and_motorcycle_detections(p_boxes, p_scores, m_boxes, m_scores),
    min_confidence=PEEING_DETECTION_CONFIDENCE,
)

state = peeing.update(
    frame,
    detections,
    run_yolo=True,  # sampled frame: run pose on person crops
    yolo_conf=PEEING_DETECTION_CONFIDENCE,
    timestamp_sec=frame_index / fps,
    frame_index=frame_index,
)

if state.edge_enter:
    print("peeing confirmed")
```

## Video pipeline hook

Implement:

```python
def scene_detector(frame_index: int, frame_bgr: np.ndarray, timestamp_sec: float) -> list[Detection]:
    person_boxes, person_scores = dfine_model.detect_persons(frame_bgr)
    moto_boxes, moto_scores = dfine_model.detect_motorcycles(frame_bgr)  # required head/classes
    return prepare_scene_detections(
        merge_person_and_motorcycle_detections(
            person_boxes, person_scores,
            moto_boxes, moto_scores,
        ),
        min_confidence=PEEING_DETECTION_CONFIDENCE,
    )
```

Import `merge_person_and_motorcycle_detections` from `models.detection_contract`.

Then:

```python
from pipelines.peeing_pipeline import load_peeing_models, run_peeing_pipeline_video, PeeingPipelineOptions

bundle = load_peeing_models()
rec = run_peeing_pipeline_video(
    bundle,
    "in.mp4",
    "out_peeing.mp4",
    scene_detector=scene_detector,
    per_video_times_init_sec=0.0,
    models_init_sec=bundle.init_sec,
    pipeline_options=PeeingPipelineOptions(),
)
```

See `examples/dfine_peeing_example.py` for a skeleton.

## DFINE-specific notes

1. **Coordinate space** — If DFINE returns normalized `[0,1]` boxes, multiply by `W` and `H`. If it returns center format, convert to xyxy pixels first.
2. **Letterbox** — If you letterbox for DFINE, map boxes back to the original frame before calling `update`.
3. **Batching** — DFINE can batch stride-sampled frames inside `scene_detector`; the pipeline calls it once per sampled index in each decode window.
4. **Class id** — Map person → `label="person"`. Map motorcycle classes → `label="motorcycle"` or `label="motorbike"` (required outputs, same list as persons).
5. **Weights** — This kit only needs **`PEEING_YOLO_POSE_MODEL`** (pose TensorRT). DFINE weights live in your repo.
6. **Motorcycles are not optional** — Integrations that only pass `person` boxes will mis-detect riders as peeing. Always merge motorcycle/motorbike detections before `PeeingDetector.update`.

## Parameter name `run_yolo`

Historical name: means **“run pose on this frame’s person detections.”** It does **not** call scene YOLO. Use `True` on stride-sampled frames, `False` when reusing carried boxes.

## Settings to tune (peeing only)

In `settings.py`: `PEEING_HAND_GROIN_Y_THRESHOLD`, `PEEING_SECONDS_REQUIRED`, stillness keys, `PEEING_DETECTION_CONFIDENCE`, pose engine path — same semantics as Trash_Detection.
