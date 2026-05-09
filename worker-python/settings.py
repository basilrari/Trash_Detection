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

RF-DETR: the video pipeline uses **TensorRT** engines only — ``weights/trash.engine`` and
``weights/cigarette.engine`` (``TRASH_ENGINE_PATH`` / ``CIGARETTE_ENGINE_PATH``) and
``TRASH_CONFIDENCE``. Requires ``tensorrt`` and ``pycuda``. Static batch and input size are
fixed inside each engine (see ``models/rfdetr_trt_trash.py``).

Trash + cigarette TensorRT heads always decode **in parallel** (two threads; shared preprocess).

Optional: ``RF_DETR_TRT_TIMING`` — set to ``1`` / ``true`` to print per-batch ``[TRT]`` timing from
``models/rfdetr_trt_trash.py`` (preprocess when measured by caller, forward, postprocess).

Optional: ``RF_DETR_PREPROCESS_CUDA`` — **default: PyTorch CUDA preprocess** when CUDA is available.
Set ``cpu`` / ``0`` / ``false`` / ``off`` / ``no`` to force the NumPy + OpenCV CPU path.

Optional path overrides: ``TRASH_ENGINE_PATH``, ``CIGARETTE_ENGINE_PATH`` (same names as settings).

Optional: ``OUTPUT_VIDEO_ENCODER`` / ``FFMPEG_PATH`` / ``NVENC_PRESET`` / ``NVENC_CQ`` for annotated MP4
(``auto`` uses ``ffmpeg`` ``h264_nvenc`` when the encoder is available, else OpenCV ``mp4v``).
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

# --- Annotated output encoding (``pipelines.test_pipeline``) ---
# OUTPUT_VIDEO_ENCODER: ``auto`` (try ``h264_nvenc`` via ffmpeg, else OpenCV ``mp4v``),
# ``nvenc`` (ffmpeg only; fails fast if unavailable), ``mp4v`` (OpenCV only).
OUTPUT_VIDEO_ENCODER = os.getenv("OUTPUT_VIDEO_ENCODER", "auto").strip().lower()
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
# NVENC preset p1…p7 (faster…slower); CQ lower = higher quality (typical 18–28).
NVENC_PRESET = os.getenv("NVENC_PRESET", "p4").strip().lower()
NVENC_CQ = int(os.getenv("NVENC_CQ", "28"))

CHUNK_SECONDS = 5
YOLO_CONFIDENCE = 0.5
PLATE_CONFIDENCE = 0.5

# --- Gating (see module docstring above) ---
# GATE_MODE: "off" | "yolo" (default: yolo — coarse/dense YOLO stride)
GATE_MODE = os.getenv("GATE_MODE", "yolo").strip().lower()
YOLO_COARSE_STRIDE = int(os.getenv("YOLO_COARSE_STRIDE", "8"))
YOLO_DENSE_STRIDE = int(os.getenv("YOLO_DENSE_STRIDE", "2"))
YOLO_DENSE_IDLE_MISS_STREAK = max(1, int(os.getenv("YOLO_DENSE_IDLE_MISS_STREAK", "10")))

# --- RF-DETR trash / cigarette (TensorRT engines only) ---
_w = Path(__file__).resolve().parent / "weights"
TRASH_ENGINE_PATH = os.getenv("TRASH_ENGINE_PATH", str(_w / "trash.engine"))
CIGARETTE_ENGINE_PATH = os.getenv("CIGARETTE_ENGINE_PATH", str(_w / "cigarette.engine"))
TRASH_CONFIDENCE = float(os.getenv("TRASH_CONFIDENCE", "0.4"))

# --- PaddleOCR (see ``models/ocr.py``) ---
# ``PADDLE_OCR_DEVICE``: unset → use GPU if Paddle sees CUDA, else CPU. Override with ``cpu``,
# ``gpu``, or ``gpu:0`` as needed.
# ``PADDLE_OCR_ISOLATE_PROCESS``: ``1``/``0`` override. On Blackwell + GPU OCR, default is
# isolated subprocess mode to avoid CUDA context collisions with torch/TRT in one process.

# --- Peeing heuristic (MediaPipe Pose on YOLO person crops; always on) ---
PEEING_POSE_STRIDE = max(1, int(os.getenv("PEEING_POSE_STRIDE", "2")))
PEEING_CROP_MARGIN = float(os.getenv("PEEING_CROP_MARGIN", "0.12"))
PEEING_MIN_VISIBILITY = float(os.getenv("PEEING_MIN_VISIBILITY", "0.45"))
PEEING_GROIN_DIST_MAX = float(os.getenv("PEEING_GROIN_DIST_MAX", "0.145"))
PEEING_GROIN_LOOSE_FACTOR = float(os.getenv("PEEING_GROIN_LOOSE_FACTOR", "1.28"))
PEEING_WRIST_BAND_MIN_VISIBILITY = float(os.getenv("PEEING_WRIST_BAND_MIN_VISIBILITY", "0.44"))
PEEING_PELVIC_BAND_Y_ABOVE = float(os.getenv("PEEING_PELVIC_BAND_Y_ABOVE", "-0.06"))
PEEING_PELVIC_BAND_Y_BELOW = float(os.getenv("PEEING_PELVIC_BAND_Y_BELOW", "0.17"))
PEEING_STANDING_Y_MARGIN = float(os.getenv("PEEING_STANDING_Y_MARGIN", "0.03"))
# Sliding window (seconds) of pose samples; alarm needs enough history in this span.
PEEING_WINDOW_SEC = float(
    os.getenv("PEEING_WINDOW_SEC", os.getenv("PEEING_MIN_ACTIVE_DURATION_SEC", "5.0"))
)
# Instant pose score on a sample counts as a "match" if >= this (0–1).
PEEING_POSE_MATCH_THRESHOLD = float(os.getenv("PEEING_POSE_MATCH_THRESHOLD", "0.6"))
# Sliding-window alarm hysteresis: enter when hit fraction **>** this (with min samples below).
PEEING_ALARM_ENTER_HIT_FRACTION = float(
    os.getenv("PEEING_ALARM_ENTER_HIT_FRACTION", "0.65")
)
# While armed, stay armed until hit fraction drops **below** this (>= holds on at equality).
PEEING_ALARM_EXIT_HIT_FRACTION = float(
    os.getenv("PEEING_ALARM_EXIT_HIT_FRACTION", "0.45")
)
# Minimum pose samples in a full-length window before the enter rule may arm.
PEEING_ALARM_MIN_SAMPLES = max(1, int(os.getenv("PEEING_ALARM_MIN_SAMPLES", "13")))
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
