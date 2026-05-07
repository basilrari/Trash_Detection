# worker-python/settings.py
"""
Defaults for the local video pipeline.

Gating (YOLO stride) — only used when ``GATE_MODE=yolo`` (see ``core/yolo_stride_gate.py``):

  * **Gate** — Short for “when do we run the expensive stack (YOLO + LP + OCR)?”
    ``GATE_MODE=off`` runs YOLO on every frame inside each time **chunk** (``CHUNK_SECONDS``).
    ``GATE_MODE=yolo`` runs YOLO on a **schedule**: mostly sparse, sometimes dense.

  * **YOLO_COARSE_STRIDE** — While the scene is treated as “idle” (no recent person/vehicle
    on a YOLO pass), run YOLO at most once every **N** frames (e.g. 8 ≈ one check every 8
    frames). Larger N = cheaper, but easier to miss very short events between samples.

  * **YOLO_DENSE_STRIDE** — After a person or vehicle is seen above ``YOLO_CONFIDENCE``,
    we open a **dense window**: for the next ``YOLO_DENSE_WINDOW_SEC`` seconds of video we
    still skip some frames, but less aggressively — run YOLO every **M** frames here
    (``M=2`` means every **other** frame: 0, 2, 4, …).

  * **YOLO_DENSE_WINDOW_SEC** — How long (in **seconds** of timeline, converted to frames
    using FPS) the dense schedule stays active after a hit. When it expires without new
    hits extending it, we fall back to coarse-only until the next person/vehicle.

Environment variables override these values when set (same names as the variables).

RF-DETR trash (``TRASH_ENABLED``, ``TRASH_WEIGHTS_PATH``, ``CIGARETTE_WEIGHTS_PATH``,
``TRASH_CONFIDENCE``, ``RF_DETR_SIZE``): optional second detector; install ``rfdetr``.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Default folders (paths relative to worker-python/ when you run from there)
INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"

# Put test videos in inputs/ (e.g. inputs/Test.mp4). Annotated results go under outputs/ by default.
VIDEO_PATH = os.getenv("VIDEO_PATH", f"{INPUTS_DIR}/Test.mp4")
OUTPUT_VIDEO = os.getenv("OUTPUT_VIDEO", f"{OUTPUTS_DIR}/annotated.mp4")

CHUNK_SECONDS = 5
YOLO_CONFIDENCE = 0.5
PLATE_CONFIDENCE = 0.5

# --- Gating (see module docstring above) ---
# GATE_MODE: "off" | "yolo"
GATE_MODE = os.getenv("GATE_MODE", "off").strip().lower()
YOLO_COARSE_STRIDE = int(os.getenv("YOLO_COARSE_STRIDE", "8"))
YOLO_DENSE_STRIDE = int(os.getenv("YOLO_DENSE_STRIDE", "2"))
YOLO_DENSE_WINDOW_SEC = float(os.getenv("YOLO_DENSE_WINDOW_SEC", "4.0"))

# --- RF-DETR trash / cigarette (optional; requires ``pip install rfdetr``) ---
TRASH_ENABLED = os.getenv("TRASH_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")
TRASH_WEIGHTS_PATH = os.getenv("TRASH_WEIGHTS_PATH", str(Path(__file__).resolve().parent / "weights" / "trash.pth"))
CIGARETTE_WEIGHTS_PATH = os.getenv(
    "CIGARETTE_WEIGHTS_PATH", str(Path(__file__).resolve().parent / "weights" / "cigarette.pth")
)
TRASH_CONFIDENCE = float(os.getenv("TRASH_CONFIDENCE", "0.4"))
# nano | small | medium | large — must match how checkpoints were trained
RF_DETR_SIZE = os.getenv("RF_DETR_SIZE", "medium").strip().lower()

LP_MODEL_PATH = "path/to/lp_model.pt"
RFDETR_MODEL_PATH = "path/to/rfdetr.pt"
