Trash Detection, Behavior Detection & License Plate Analysis System

A modular, GPU-accelerated, multi-model video analysis pipeline (Python only).

This repository analyzes CCTV-style videos in chunks using computer-vision models. The layout is structured so models and pipelines can be swapped or extended safely.

## What runs today

The **Python** package under `worker-python/` streams a video, runs **YOLO** (people/vehicles), **license-plate YOLO** on vehicle crops, **PaddleOCR** on plate crops, and writes an **annotated MP4**.

Planned extensions (from the original design): pose/behavior, litter (e.g. RFDETR), CSV export — not all are wired yet.

## Running

From `worker-python/` (with your GPU conda/env and weights in `worker-python/weights/`):

```bash
# Defaults: VIDEO_PATH and OUTPUT_VIDEO in settings.py (or env VIDEO_PATH / OUTPUT_VIDEO)
python worker.py

# Or pass paths on the command line
python worker.py /path/to/video.mp4 -o /path/to/annotated.mp4
```

Alternatively, using defaults from `settings.py` only:

```bash
python -m pipelines.test_pipeline
```

(requires running with `worker-python` as the current working directory so `models` / `settings` imports resolve.)

## Layout

- `worker.py` — CLI entrypoint
- `settings.py` — default video paths, chunk size, confidence thresholds, optional `.env` via `python-dotenv` if installed
- `pipelines/test_pipeline.py` — `run_pipeline(video_path, output_video)`
- `models/` — YOLO, LP detector, OCR wrappers; `base.py` interfaces
- `core/` — shared types and CSV helper (`writer.py`) for future use
- `weights/` — place `yolo11x.pt`, `bestlicense.pt` here (see `.gitignore`)

## Model weights and GPU

- YOLO: `worker-python/weights/yolo11x.pt`
- License plates: `worker-python/weights/bestlicense.pt`
- PaddleOCR is configured for GPU in `models/ocr.py`
