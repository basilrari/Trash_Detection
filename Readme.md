Trash Detection, Behavior Detection & License Plate Analysis System

A modular, GPU-accelerated, multi-model video analysis pipeline (Python only).

This repository analyzes CCTV-style videos using computer-vision models. The layout is structured so models and pipelines can be swapped or extended safely.

## What runs today

The **Python** package under `worker-python/` streams a video, runs **scene YOLO** on a **uniform stride** (`FRAME_SAMPLE_STRIDE` in `settings.py` — every *N*th decoded frame, micro-batched with `YOLO_MICRO_BATCH_SIZE`), **RF-DETR** litter heads via **TensorRT** (paths from `TRASH_ENGINE_PATH` / `CIGARETTE_ENGINE_PATH` in `settings.py`), **license-plate YOLO** on vehicle crops, **PaddleOCR** on plate crops, and writes an **annotated MP4**.

Production inputs are expected at **10–60 FPS** (nominal container FPS). Values outside that range still run but **log a warning** when the file is opened.

Planned extensions (from the original design): pose/behavior, CSV export — not all are wired yet.

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

### Scene YOLO sampling (`FRAME_SAMPLE_STRIDE`)

**Scene YOLO** runs only on decoded frames where `frame_index % FRAME_SAMPLE_STRIDE == 0`. Other frames reuse the **last sampled** scene boxes (peeing carry, cached plate redraw). **RF-DETR** runs on sampled frames that have person/vehicle activity at `YOLO_CONFIDENCE`. **LP/OCR** follow the same sampling rules as in `pipelines/test_pipeline.py`.

| Setting | Meaning |
|---------|--------|
| **`FRAME_SAMPLE_STRIDE`** | Integer ≥ **1**. Scene YOLO on indices `0, N, 2N, …`. With nominal FPS in **[10, 60]**, stride `N` implies about **FPS/N** scene-YOLO evaluations per second. |
| **`YOLO_MICRO_BATCH_SIZE`** | Batched `detect()` calls: up to this many **sampled** frames per launch. |
| **`INPUT_VIDEO_FPS_MIN`** / **`INPUT_VIDEO_FPS_MAX`** | **10** / **60** — expected nominal input FPS; **warning only** if OpenCV reports FPS outside this range. |

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
