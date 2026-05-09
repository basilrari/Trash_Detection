Trash Detection, Behavior Detection & License Plate Analysis System

A modular, GPU-accelerated, multi-model video analysis pipeline (Python only).

This repository analyzes CCTV-style videos in chunks using computer-vision models. The layout is structured so models and pipelines can be swapped or extended safely.

## What runs today

The **Python** package under `worker-python/` streams a video, runs **YOLO** with a **YOLO stride gate by default** (`GATE_MODE=yolo`), **RF-DETR** litter heads via **TensorRT** (`weights/trash.engine` and `weights/cigarette.engine`), **license-plate YOLO** on vehicle crops, **PaddleOCR** on plate crops, and writes an **annotated MP4**.

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

# CLI: video path and -o only; gating is configured in settings.py (see below)
python worker.py --help
```

### Gating (`GATE_MODE`)

The **gate** is the rule that decides **how often** we run **scene YOLO** (and therefore LP/OCR, which need YOLO boxes). **RF-DETR** follows scene activity on frames where YOLO ran (see `GATE_MODE` table below).

| Setting / env | Meaning |
|---------------|--------|
| **`GATE_MODE`** | **`yolo`**: coarse / dense YOLO stride schedule (see `core/yolo_stride_gate.py`). **`off`**: time-chunk path — YOLO on every decoded frame in each chunk (`CHUNK_SECONDS`). **`"1"`**, **`"2"`**, …: uniform stride — scene YOLO only on frames where `index % N == 0`, micro-batched; skipped frames reuse the last scene boxes (see `pipelines/test_pipeline.py`). |
| **`YOLO_COARSE_STRIDE`** | When **idle** (not in dense mode), run YOLO only on frames where the index is a multiple of this value (e.g. **8** → frames 0, 8, 16, …). Larger = cheaper sampling; typical **5–10**. |
| **`YOLO_DENSE_STRIDE`** | While **dense mode** is on (after YOLO sees a person or vehicle above `YOLO_CONFIDENCE`), also run YOLO every **this many** frames. **`2`** = every **other** frame (among dense-scheduled indices). |
| **`YOLO_DENSE_IDLE_MISS_STREAK`** | Dense mode is **not** a fixed time window. After a hit, we stay dense until YOLO has run **this many times in a row** with **no** person/vehicle (default **10**). Then we return to coarse-only until the next hit. |

Edit **`worker-python/settings.py`** (`GATE_MODE`, `YOLO_*` strides, `CHUNK_SECONDS`, etc.) — the app does not read shell environment for these.

```bash
python worker.py inputs/myvideo.mp4 -o outputs/out.mp4
```

In code, see the module docstring at the top of **`worker-python/settings.py`**, **`worker-python/core/yolo_stride_gate.py`**, and the docstring at the top of **`worker-python/pipelines/test_pipeline.py`**.

### RF-DETR trash (required)

Inference uses **TensorRT** only: place **`trash.engine`** and **`cigarette.engine`** under `worker-python/weights/` (or set **`TRASH_ENGINE_PATH`** and **`CIGARETTE_ENGINE_PATH`**). You need **`tensorrt`** and **`pycuda`**. Engine batch size and input resolution are fixed inside each plan file (see `models/rfdetr_trt_trash.py`). Each batch is **preprocessed once** (shared NCHW input); trash and cigarette heads then run TensorRT **decode in parallel** (two threads). Optional archival PyTorch checkpoints **`trash.pth`** / **`cigarette.pth`** may live alongside them but are **not** loaded by this pipeline.

| Setting / env | Meaning |
|---------------|--------|
| **`TRASH_ENGINE_PATH`** | Path to `trash.engine` (default: `weights/trash.engine`). |
| **`CIGARETTE_ENGINE_PATH`** | Path to `cigarette.engine` (default: `weights/cigarette.engine`). **Required.** |
| **`TRASH_CONFIDENCE`** | Confidence threshold for trash boxes (default **0.4**). |

With **`GATE_MODE=yolo`**, trash runs on the same cadence as YOLO (coarse/dense). With **`GATE_MODE=off`**, trash runs on frames in each chunk that have scene activity. With a **numeric** `GATE_MODE`, trash runs only on **sampled** frames that have activity.

Alternatively, using defaults from `settings.py` only:

```bash
python -m pipelines.test_pipeline
```

(requires running with `worker-python` as the current working directory so `models` / `settings` imports resolve.)

## Layout

- `inputs/` — drop test / production clips here (tracked empty via `.gitkeep`; media files are gitignored)
- `outputs/` — annotated videos written here by default
- `worker.py` — CLI entrypoint
- `settings.py` — default video paths, chunk size, confidence thresholds, optional `.env` via `python-dotenv` if installed
- `pipelines/test_pipeline.py` — `run_pipeline(video_path, output_video)`
- `models/` — YOLO, LP detector, OCR, RF-DETR TensorRT (`rfdetr_trt_trash.py`, `trash_detector.py` helpers); `base.py` interfaces
- `core/` — `types.py`, `writer.py`, `yolo_stride_gate.py` (YOLO coarse/dense scheduling; default on)
- `weights/` — place `yolo11x.pt`, `bestlicense.pt`, **`trash.engine`** and **`cigarette.engine`** (RF-DETR); optional `trash.pth` / `cigarette.pth` for archival only (see `.gitignore`)

## Model weights and GPU

- YOLO: `worker-python/weights/yolo11x.pt`
- License plates: `worker-python/weights/bestlicense.pt`
- RF-DETR: `worker-python/weights/trash.engine` and `worker-python/weights/cigarette.engine` (optional `trash.pth` / `cigarette.pth` not used at runtime)
- PaddleOCR is configured for GPU in `models/ocr.py`
