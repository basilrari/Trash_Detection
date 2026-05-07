# worker-python/settings.py
"""
Defaults for the local video pipeline.

Gating (YOLO stride) — default ``GATE_MODE=yolo`` (see ``core/yolo_stride_gate.py``):

  * **Gate** — Short for “when do we run the expensive stack (YOLO + LP + OCR)?”
    ``GATE_MODE=yolo`` (default) runs YOLO on a **schedule**: mostly sparse, sometimes dense.
    ``GATE_MODE=off`` runs YOLO on every frame inside each time **chunk** (``CHUNK_SECONDS``).

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

RF-DETR trash is **required** for the pipeline: install ``rfdetr``, place ``weights/trash.pth``
(``TRASH_WEIGHTS_PATH``), optional ``cigarette.pth``, and set ``TRASH_CONFIDENCE`` /
``RF_DETR_SIZE`` to match your checkpoints.

Fine-tuned checkpoints often store training ``args`` (``patch_size``, ``resolution``,
``num_classes``, …). Those are merged into the RF-DETR constructor when compatible with
the chosen size class. If ``args`` omit backbone geometry, ``positional_encoding_size`` and
``resolution`` are inferred from the saved ``position_embeddings`` tensor when the patch grid
is square. Optional overrides: ``RF_DETR_PATCH_SIZE``, ``RF_DETR_NUM_CLASSES``,
``RF_DETR_RESOLUTION``, ``RF_DETR_POSITIONAL_ENCODING_SIZE`` (integers; unset = do not override).
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
# GATE_MODE: "off" | "yolo" (default: yolo — coarse/dense YOLO stride)
GATE_MODE = os.getenv("GATE_MODE", "yolo").strip().lower()
YOLO_COARSE_STRIDE = int(os.getenv("YOLO_COARSE_STRIDE", "8"))
YOLO_DENSE_STRIDE = int(os.getenv("YOLO_DENSE_STRIDE", "2"))
YOLO_DENSE_WINDOW_SEC = float(os.getenv("YOLO_DENSE_WINDOW_SEC", "4.0"))

# --- RF-DETR trash / cigarette (required; ``pip install rfdetr``) ---
TRASH_WEIGHTS_PATH = os.getenv("TRASH_WEIGHTS_PATH", str(Path(__file__).resolve().parent / "weights" / "trash.pth"))
CIGARETTE_WEIGHTS_PATH = os.getenv(
    "CIGARETTE_WEIGHTS_PATH", str(Path(__file__).resolve().parent / "weights" / "cigarette.pth")
)
TRASH_CONFIDENCE = float(os.getenv("TRASH_CONFIDENCE", "0.4"))
# nano | small | medium | large — backbone family; checkpoint ``args`` still tune patch_size etc.
RF_DETR_SIZE = os.getenv("RF_DETR_SIZE", "medium").strip().lower()


def _optional_int_env(var: str) -> int | None:
    raw = os.getenv(var, "").strip()
    return int(raw) if raw else None


# Optional RF-DETR ctor overrides (leave unset to use checkpoint ``args`` merge + library defaults)
RF_DETR_PATCH_SIZE = _optional_int_env("RF_DETR_PATCH_SIZE")
RF_DETR_NUM_CLASSES = _optional_int_env("RF_DETR_NUM_CLASSES")
RF_DETR_RESOLUTION = _optional_int_env("RF_DETR_RESOLUTION")
RF_DETR_POSITIONAL_ENCODING_SIZE = _optional_int_env("RF_DETR_POSITIONAL_ENCODING_SIZE")

LP_MODEL_PATH = "path/to/lp_model.pt"
RFDETR_MODEL_PATH = "path/to/rfdetr.pt"
