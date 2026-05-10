Trash Detection, Behavior Detection & License Plate Analysis System

A modular, GPU-accelerated, multi-model video analysis pipeline (Python only).

This repository analyzes CCTV-style videos using computer-vision models. The layout is structured so models and pipelines can be swapped or extended safely.

## What runs today

The **Python** package under `worker-python/` streams a video, runs **scene YOLO** on a **uniform stride**
(stride is **automatic**: target **`SCENE_YOLO_TARGET_FRAMES_PER_SECOND`** scene-YOLO frames per **second of video**
from reported FPS—see `settings.py`, unless **`FRAME_SAMPLE_STRIDE_OVERRIDE`** is set), micro-batched with `YOLO_MICRO_BATCH_SIZE`, **RF-DETR** litter heads via **TensorRT** (paths from `TRASH_ENGINE_PATH` / `CIGARETTE_ENGINE_PATH` in `settings.py`), **license-plate YOLO** on vehicle crops, **PaddleOCR** on plate crops, and writes an **annotated MP4**.

Production inputs are expected at **5–60 FPS** (nominal container FPS). Values outside that range still run but **log a warning** when the file is opened.

Planned extensions (from the original design): CSV export and further behavior rules — not all are wired yet.

**Peeing cue** (always on): default **YOLO pose** on **scene-YOLO person** crops from a **TensorRT** ``.engine`` only (**`PEEING_YOLO_POSE_MODEL`**), **stride-aligned** with scene YOLO. Rule: standing + hand near groin; per tracked person, **≥ `PEEING_MIN_HITS_PER_SECOND`** pose hits in a **calendar second**, repeated **`PEEING_SECONDS_REQUIRED`** consecutive seconds → confirm. Tunables: `PEEING_*` in `settings.py` (IoU tracking, crop margin, **`PEEING_POSE_BACKEND`** `yolo`/`mediapipe`, **`PEEING_MEDIAPIPE_MODE`**, optional **`PEEING_MAX_POSE_PERSONS_PER_FRAME`**). **Overlays:** scene YOLO and LP are **label-only**; RF-DETR **trash** / **cigarette** use **boxed** annotations (red / yellow by class); peeing cues use **thick red** boxes on sampled frames. **`PEEING_DEBUG_TIMING`:** prints counters to **stderr** when the detector closes.

## Running

From `worker-python/` (with your GPU conda/env and weights in `worker-python/weights/`).

Put source videos in **`worker-python/inputs/`** (for example `inputs/Test.mp4`). Annotated results are written under **`worker-python/outputs/`** by default (`outputs/annotated.mp4`, or `outputs/<name>_annotated.mp4` when you pass a file on the CLI).

```bash
# Defaults: VIDEO_PATH / OUTPUT_VIDEO in settings.py (default inputs/Test.mp4 → outputs/annotated.mp4)
python worker.py

# Or pass paths (still use inputs/ and outputs/ as a convention)
python worker.py inputs/myvideo.mp4
python worker.py inputs/myvideo.mp4 -o outputs/custom.mp4

# CLI: video path and -o only; stride and thresholds are configured in settings.py (see below)
python worker.py --help
```

### Scene YOLO stride (automatic + optional override)

**Scene YOLO** runs only on decoded frames where `frame_index % stride == 0`. The effective **stride** is:

- **`FRAME_SAMPLE_STRIDE_OVERRIDE`** — if set to an integer ≥ **1**, that stride is used (fixed).
- Otherwise **automatic:** `stride = max(1, round(fps_for_stride / SCENE_YOLO_TARGET_FRAMES_PER_SECOND))`, where **`fps_for_stride`** is the reported FPS **clamped** to **`[INPUT_VIDEO_FPS_MIN, INPUT_VIDEO_FPS_MAX]`**. Default **`SCENE_YOLO_TARGET_FRAMES_PER_SECOND = 5`** ⇒ at **10 FPS** stride **2** (every other frame); at **60 FPS** stride **12** (five scene-YOLO frames each second of video). Integer stride cannot hit exactly five for every FPS; the realized rate stays **near** the target.

Approximate scene-YOLO frames per second of video ≈ **`fps_for_stride / stride`** (≈ **`SCENE_YOLO_TARGET_FRAMES_PER_SECOND`** when FPS divides cleanly). Other frames reuse the **last sampled** scene boxes. **RF-DETR** runs on sampled frames that have person/vehicle activity at `YOLO_CONFIDENCE`.

| Setting | Meaning |
|---------|--------|
| **`SCENE_YOLO_TARGET_FRAMES_PER_SECOND`** | Target scene-YOLO runs **per second of video** when stride is automatic (default **5**). |
| **`FRAME_SAMPLE_STRIDE_OVERRIDE`** | **`None`** for automatic stride; or an integer ≥ **1** to force a fixed stride. |
| **`YOLO_MICRO_BATCH_SIZE`** | Batched `detect()` calls: up to this many **sampled** frames per launch. |
| **`INPUT_VIDEO_FPS_MIN`** / **`INPUT_VIDEO_FPS_MAX`** | **5** / **60** — nominal FPS band; **warning** if OpenCV reports outside it; also used to **clamp** FPS for automatic stride. |

Edit **`worker-python/settings.py`** — literals only; the app does **not** read these from the shell environment.

```bash
python worker.py inputs/myvideo.mp4 -o outputs/out.mp4
```

In code, see the module docstring at the top of **`worker-python/settings.py`** and **`worker-python/pipelines/test_pipeline.py`**.

### RF-DETR trash (required)

Inference uses **TensorRT**: set **`TRASH_ENGINE_PATH`** and **`CIGARETTE_ENGINE_PATH`** in `settings.py` (defaults under `worker-python/weights/`). You need **`tensorrt`** and **`pycuda`**. Engine batch size and input resolution are fixed inside each plan file (see `models/rfdetr_trt_trash.py`). Each batch is **preprocessed once** (shared NCHW input); trash and cigarette heads then run TensorRT **decode in parallel** (two threads). Optional archival PyTorch checkpoints **`trash.pth`** / **`cigarette.pth`** may live alongside them but are **not** loaded by this pipeline.

| Setting | Meaning |
|---------|--------|
| **`TRASH_ENGINE_PATH`** | TensorRT plan for the trash head (see `settings.py` default under `weights/`). |
| **`CIGARETTE_ENGINE_PATH`** | TensorRT plan for the cigarette head. **Required.** |
| **`TRASH_CONFIDENCE`** | Confidence threshold for trash boxes (default **0.4**). |

Trash inference follows **scene YOLO** cadence: it runs on **sampled** frames where scene activity meets **`YOLO_CONFIDENCE`**, using batched RF-DETR as implemented in `pipelines/test_pipeline.py`.

Alternatively, using defaults from `settings.py` only:

```bash
python -m pipelines.test_pipeline
```

(requires running with `worker-python` as the current working directory so `models` / `settings` imports resolve.)

## Layout

- `inputs/` — drop test / production clips here (tracked empty via `.gitkeep`; media files are gitignored)
- `outputs/` — annotated videos written here by default
- `worker.py` — CLI entrypoint
- `settings.py` — default video paths, stride, confidence thresholds, encoder options (edit literals in file)
- `pipelines/test_pipeline.py` — `run_pipeline(video_path, output_video)`
- `models/` — shared **`types.py`**, YOLO, LP detector, OCR, RF-DETR TensorRT (`rfdetr_trt_trash.py`, `trash_detector.py` helpers); `base.py` interfaces
- `weights/` — scene/LP **TensorRT** ``.engine`` files and RF-DETR engines per `settings.py`; optional archival `.pth` can live under `weights/old models/` (see `.gitignore`)

## Model weights and GPU

- Scene YOLO / LP: **`YOLO_ENGINE_PATH`** and **`LP_ENGINE_PATH`** in **`worker-python/settings.py`** (defaults under `weights/`)
- RF-DETR: engine paths from **`TRASH_ENGINE_PATH`** and **`CIGARETTE_ENGINE_PATH`** (optional `trash.pth` / `cigarette.pth` not used at runtime)
- PaddleOCR is configured for GPU in `models/ocr.py`
