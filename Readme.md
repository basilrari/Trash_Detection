Trash Detection, Behavior Detection & License Plate Analysis System

A modular, GPU-accelerated, multi-model video analysis pipeline (Python only).

This repository analyzes CCTV-style videos in chunks using computer-vision models. The layout is structured so models and pipelines can be swapped or extended safely.

## What runs today

The **Python** package under `worker-python/` streams a video, runs **YOLO** with a **YOLO stride gate by default** (`GATE_MODE=yolo`), **RF-DETR** litter heads (required: `rfdetr` + `weights/trash.pth`), **license-plate YOLO** on vehicle crops, **PaddleOCR** on plate crops, and writes an **annotated MP4**.

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

# CLI help includes gating flags (--gate, --yolo-coarse-stride, …)
python worker.py --help
```

### Gating (`GATE_MODE`)

The **gate** is the rule that decides **how often** we run **YOLO** (and therefore LP/OCR, which need YOLO boxes). **RF-DETR** follows the same schedule as YOLO when `GATE_MODE=yolo`.

| Setting / env | Meaning |
|---------------|--------|
| **`GATE_MODE`** | **`yolo`** (default): coarse / dense YOLO stride schedule (see `core/yolo_stride_gate.py`). **`off`**: no stride gate — within each time **chunk** (`CHUNK_SECONDS` in `settings.py`), YOLO runs on **every** frame in that chunk; RF-DETR runs on every frame in each chunk as well. |
| **`YOLO_COARSE_STRIDE`** | When we are **not** in a “dense attention” window, run YOLO only on frames where the index is a multiple of this value (e.g. **8** → frames 0, 8, 16, …). Larger = cheaper idle sampling; typical range **5–10**. |
| **`YOLO_DENSE_STRIDE`** | After YOLO sees a **person** or **vehicle** (above `YOLO_CONFIDENCE`), we open a **dense window**: run YOLO every **this many** frames while that window is active. **`2`** = every **other** frame. |
| **`YOLO_DENSE_WINDOW_SEC`** | How many **seconds** of video (timeline) the dense schedule stays on after a hit; converted to frames with FPS. New hits **extend** the window. |

```bash
# Use full YOLO + RF-DETR on every frame inside each chunk (no stride gate)
export GATE_MODE=off
python worker.py inputs/myvideo.mp4 -o outputs/out.mp4

# Tune the default yolo gate for this run
export GATE_MODE=yolo
export YOLO_COARSE_STRIDE=10                             # idle: YOLO at most every 10th frame
export YOLO_DENSE_STRIDE=2                              # when busy: YOLO every 2nd frame (0,2,4,…)
export YOLO_DENSE_WINDOW_SEC=5                           # stay “dense” for ~5s after a person/vehicle hit
python worker.py inputs/myvideo.mp4 -o outputs/out.mp4

# Same tuning via CLI flags (no exports); overrides apply before settings are read
python worker.py --gate yolo \
  --yolo-coarse-stride 10 \
  --yolo-dense-stride 2 \
  --yolo-dense-window-sec 5 \
  inputs/myvideo.mp4 -o outputs/out.mp4
```

In code, see the module docstring at the top of **`worker-python/settings.py`**, **`worker-python/core/yolo_stride_gate.py`**, and the docstring at the top of **`worker-python/pipelines/test_pipeline.py`**.

### RF-DETR trash (required)

**RF-DETR is required** for `worker.py` / `run_pipeline`: install `rfdetr` (`worker-python/requirements.txt`), place **`trash.pth`** under `worker-python/weights/` (or set `TRASH_WEIGHTS_PATH`). Optional second head: `cigarette.pth`. **`RF_DETR_SIZE`** must match how those weights were trained (`nano` \| `small` \| `medium` \| `large`). If `rfdetr` or weights are missing, the process exits with an error instead of skipping trash.

| Setting / env | Meaning |
|---------------|--------|
| **`TRASH_WEIGHTS_PATH`** | Path to `trash.pth` (default: `weights/trash.pth`). |
| **`CIGARETTE_WEIGHTS_PATH`** | Optional second head; merged if the file exists and differs from trash weights. |
| **`TRASH_CONFIDENCE`** | Confidence threshold for trash boxes (default **0.4**). |
| **`RF_DETR_SIZE`** | Model backbone size; must match checkpoints. |

With **`GATE_MODE=yolo`** (default), trash runs on the same cadence as YOLO (coarse/dense). With **`GATE_MODE=off`**, trash runs on every frame in each chunk.

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
- `models/` — YOLO, LP detector, OCR, RF-DETR trash (`trash_detector.py`); `base.py` interfaces
- `core/` — `types.py`, `writer.py`, `yolo_stride_gate.py` (YOLO coarse/dense scheduling; default on)
- `weights/` — place `yolo11x.pt`, `bestlicense.pt`, **`trash.pth`** (required), optional `cigarette.pth` (see `.gitignore`)

## Model weights and GPU

- YOLO: `worker-python/weights/yolo11x.pt`
- License plates: `worker-python/weights/bestlicense.pt`
- PaddleOCR is configured for GPU in `models/ocr.py`
