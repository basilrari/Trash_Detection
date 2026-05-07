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

  * **YOLO_DENSE_STRIDE** — While **dense mode** is on (after YOLO sees a person or vehicle
    above ``YOLO_CONFIDENCE``), also run YOLO every **M** frames (``M=2`` → every other frame
    among indices that satisfy the dense rule, in addition to coarse samples).

  * **YOLO_DENSE_IDLE_MISS_STREAK** — Dense mode is **not** a fixed time window. After a hit,
    we stay dense until YOLO has run **this many times in a row** with **no** person/vehicle
    (default **10**). Then we return to coarse-only until the next hit.

Environment variables override these values when set (same names as the variables).

RF-DETR is **required**: install ``rfdetr``, place **both** ``weights/trash.pth`` and
``weights/cigarette.pth`` (``TRASH_WEIGHTS_PATH`` / ``CIGARETTE_WEIGHTS_PATH``), and set
``TRASH_CONFIDENCE``. The ``rfdetr`` model family is inferred from each checkpoint (``args``
or trial load); no separate size setting.

Fine-tuned checkpoints often store training ``args`` (``patch_size``, ``resolution``,
``num_classes``, …). Those are merged into the RF-DETR constructor when compatible with
the inferred size class. If ``args`` omit backbone geometry, ``positional_encoding_size`` and
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
YOLO_DENSE_IDLE_MISS_STREAK = max(1, int(os.getenv("YOLO_DENSE_IDLE_MISS_STREAK", "10")))

# --- RF-DETR trash / cigarette (required; ``pip install rfdetr``) ---
# Fixed local checkpoints only (no path override): worker-python/weights/{trash,cigarette}.pth
TRASH_WEIGHTS_PATH = str(Path(__file__).resolve().parent / "weights" / "trash.pth")
CIGARETTE_WEIGHTS_PATH = str(Path(__file__).resolve().parent / "weights" / "cigarette.pth")
TRASH_CONFIDENCE = float(os.getenv("TRASH_CONFIDENCE", "0.4"))


def _optional_int_env(var: str) -> int | None:
    raw = os.getenv(var, "").strip()
    return int(raw) if raw else None


# Optional RF-DETR ctor overrides (leave unset to use checkpoint ``args`` merge + library defaults)
RF_DETR_PATCH_SIZE = _optional_int_env("RF_DETR_PATCH_SIZE")
RF_DETR_NUM_CLASSES = _optional_int_env("RF_DETR_NUM_CLASSES")
RF_DETR_RESOLUTION = _optional_int_env("RF_DETR_RESOLUTION")
RF_DETR_POSITIONAL_ENCODING_SIZE = _optional_int_env("RF_DETR_POSITIONAL_ENCODING_SIZE")

# --- Peeing heuristic (MediaPipe Pose on YOLO person crops; always on) ---
PEEING_POSE_STRIDE = max(1, int(os.getenv("PEEING_POSE_STRIDE", "2")))
PEEING_CROP_MARGIN = float(os.getenv("PEEING_CROP_MARGIN", "0.12"))
PEEING_MIN_VISIBILITY = float(os.getenv("PEEING_MIN_VISIBILITY", "0.45"))
PEEING_GROIN_DIST_MAX = float(os.getenv("PEEING_GROIN_DIST_MAX", "0.13"))
PEEING_GROIN_LOOSE_FACTOR = float(os.getenv("PEEING_GROIN_LOOSE_FACTOR", "1.22"))
PEEING_WRIST_BAND_MIN_VISIBILITY = float(os.getenv("PEEING_WRIST_BAND_MIN_VISIBILITY", "0.44"))
PEEING_PELVIC_BAND_Y_ABOVE = float(os.getenv("PEEING_PELVIC_BAND_Y_ABOVE", "-0.06"))
PEEING_PELVIC_BAND_Y_BELOW = float(os.getenv("PEEING_PELVIC_BAND_Y_BELOW", "0.17"))
PEEING_STANDING_Y_MARGIN = float(os.getenv("PEEING_STANDING_Y_MARGIN", "0.03"))
PEEING_EMA_ALPHA = float(os.getenv("PEEING_EMA_ALPHA", "0.4"))
PEEING_ACTIVE_THRESHOLD = float(os.getenv("PEEING_ACTIVE_THRESHOLD", "0.62"))
# Clear pose focus boxes when EMA falls below this (softer than active_threshold).
PEEING_EMA_RELEASE_THRESHOLD = float(os.getenv("PEEING_EMA_RELEASE_THRESHOLD", "0.44"))
# Seconds of video time with EMA >= PEEING_ACTIVE_THRESHOLD before PEEING UI / edges turn on.
PEEING_MIN_ACTIVE_DURATION_SEC = float(os.getenv("PEEING_MIN_ACTIVE_DURATION_SEC", "5.0"))
PEEING_DECAY_NO_YOLO = float(os.getenv("PEEING_DECAY_NO_YOLO", "0.97"))
PEEING_DECAY_NO_PERSON = float(os.getenv("PEEING_DECAY_NO_PERSON", "0.96"))
PEEING_SQUAT_HIP_KNEE_GAP_MAX = float(os.getenv("PEEING_SQUAT_HIP_KNEE_GAP_MAX", "0.09"))
PEEING_SQUAT_DEPTH_SCALE = float(os.getenv("PEEING_SQUAT_DEPTH_SCALE", "0.11"))
PEEING_POSE_MODEL_PATH = os.getenv(
    "PEEING_POSE_MODEL_PATH",
    str(Path.home() / ".cache" / "trash_detection_worker" / "pose_landmarker_lite.task"),
)
PEEING_POSE_MODEL_URL = os.getenv(
    "PEEING_POSE_MODEL_URL",
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
)
