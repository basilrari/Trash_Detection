# Portable peeing detection kit

Peeing logic from **Trash_Detection** (pose TRT + standing / groin / stillness / temporal rules). **No scene detector included** — your repo (e.g. **DFINE**) supplies person boxes.

Current thresholds match the main project (`PEEING_HAND_GROIN_Y_THRESHOLD = 0.09`, `PEEING_SECONDS_REQUIRED = 10`, etc.) in `settings.py`.

## Start here

| Doc | Purpose |
|-----|---------|
| **[DFINE_INTEGRATION.md](DFINE_INTEGRATION.md)** | How to pass person **and motorcycle** detections into pose / peeing |
| **`models/detection_contract.py`** | `Detection` format + conversion helpers |
| **`examples/dfine_peeing_example.py`** | Video pipeline skeleton |

## Layout

```
portable_peeing/
  settings.py                 # peeing + pose + encode (no scene YOLO paths)
  models/
    peeing_detector.py        # core logic; update() takes your Detection list
    detection_contract.py     # DFINE → Detection helpers
    peeing_*.py               # gate, stillness, viz
  pipelines/
    peeing_pipeline.py        # video loop; requires scene_detector callback
  weights/                    # pose engine only (yolo11n-pose_b8_fp16.engine)
  tests/
```

## What you implement

```python
def scene_detector(frame_index: int, frame_bgr, timestamp_sec) -> list[Detection]:
    p_boxes, p_scores = your_dfine_persons(frame_bgr)
    m_boxes, m_scores = your_dfine_motorcycles(frame_bgr)  # required — not optional
    return prepare_scene_detections(
        merge_person_and_motorcycle_detections(p_boxes, p_scores, m_boxes, m_scores),
        min_confidence=PEEING_DETECTION_CONFIDENCE,
    )
```

Then:

```python
from pipelines.peeing_pipeline import load_peeing_models, run_peeing_pipeline_video

bundle = load_peeing_models()
run_peeing_pipeline_video(
    bundle, "in.mp4", "out.mp4",
    scene_detector=scene_detector,
    per_video_times_init_sec=0.0,
    models_init_sec=bundle.init_sec,
)
```

Or call **`PeeingDetector.update(frame, detections, run_yolo=True, ...)`** directly in your own loop (see DFINE_INTEGRATION.md).

## Weights

Copy only the **pose** TensorRT engine into `weights/` (see `weights/README.md`). DFINE weights stay in your repo.

## Tests

```bash
cd portable_peeing
python -m unittest discover -s tests -v
```

## Dependencies

`pip install -r requirements.txt` — OpenCV, Ultralytics, PyTorch CUDA, ffmpeg (optional NVENC).

## Trash_Detection vs this kit

| Trash_Detection `worker-python` | This kit |
|---------------------------------|----------|
| Scene YOLO TRT for person boxes | **You** run DFINE (or other) |
| Same `PeeingDetector` rules | Same rules |
| `peeing_worker.py` CLI | Use `examples/dfine_peeing_example.py` or your integration |
