# worker-python/settings.py
"""
Local video pipeline configuration.

**Single source of truth:** edit the literals in this file. Nothing is read from the OS
environment or ``.env`` for application settings.

``python worker.py`` may override ``GATE_MODE`` and YOLO stride fields in memory for that
process only when you pass ``--gate`` / ``--yolo-*`` flags (see ``worker.py``).
"""
from pathlib import Path

# --- Default folders (paths relative to worker-python/ when you run from there) ---
INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"

VIDEO_PATH = f"{INPUTS_DIR}/Test.mp4"
OUTPUT_VIDEO = f"{OUTPUTS_DIR}/annotated.mp4"

# --- Annotated output encoding (``pipelines.test_pipeline``) ---
# OUTPUT_VIDEO_ENCODER: ``auto`` (try ``h264_nvenc`` via ffmpeg, else OpenCV ``mp4v``),
# ``nvenc`` (ffmpeg only; fails fast if unavailable), ``mp4v`` (OpenCV only).
OUTPUT_VIDEO_ENCODER = "auto"
FFMPEG_PATH = "ffmpeg"
NVENC_PRESET = "p4"
NVENC_CQ = 28

CHUNK_SECONDS = 5
YOLO_CONFIDENCE = 0.5
PLATE_CONFIDENCE = 0.5

# --- Gating (``core/yolo_stride_gate.py``) ---
# GATE_MODE: "off" | "yolo" (default: yolo — coarse/dense YOLO stride)
GATE_MODE = "yolo"
YOLO_COARSE_STRIDE = 10
YOLO_DENSE_STRIDE = 2
YOLO_DENSE_IDLE_MISS_STREAK = 8

# --- RF-DETR trash / cigarette (TensorRT engines only) ---
_w = Path(__file__).resolve().parent / "weights"
TRASH_ENGINE_PATH = str(_w / "trash.engine")
CIGARETTE_ENGINE_PATH = str(_w / "cigarette.engine")
TRASH_CONFIDENCE = 0.4

# Extra ``[TRT]`` timing lines from ``models/rfdetr_trt_trash.py``.
RF_DETR_TRT_TIMING = False

# RF-DETR preprocess: use CPU (NumPy + OpenCV) unless you set a CUDA opt-in value here:
# ``"1"``, ``"true"``, ``"yes"``, ``"on"``, ``"cuda"``, ``"auto"`` (CUDA when available).
# ``""``, ``"0"``, ``"cpu"``, ``"false"``, ``"off"``, ``"no"`` → CPU.
RF_DETR_PREPROCESS_CUDA = "1"

# --- PaddleOCR (``models/ocr.py``) ---
# ``""`` → auto (GPU if Paddle sees CUDA). Otherwise ``"cpu"``, ``"gpu"``, ``"gpu:0"``, etc.
PADDLE_OCR_DEVICE = "gpu"
# ``None`` → default isolation rule (Blackwell + GPU OCR). ``True`` / ``False`` to force.
PADDLE_OCR_ISOLATE_PROCESS: bool | None = None

# --- Peeing heuristic (MediaPipe Pose on YOLO person crops; always on) ---
PEEING_POSE_STRIDE = 2
PEEING_CROP_MARGIN = 0.12
PEEING_MIN_VISIBILITY = 0.45
PEEING_GROIN_DIST_MAX = 0.145
PEEING_GROIN_LOOSE_FACTOR = 1.28
PEEING_WRIST_BAND_MIN_VISIBILITY = 0.44
PEEING_PELVIC_BAND_Y_ABOVE = -0.06
PEEING_PELVIC_BAND_Y_BELOW = 0.17
PEEING_STANDING_Y_MARGIN = 0.03
PEEING_WINDOW_SEC = 5.0
PEEING_POSE_MATCH_THRESHOLD = 0.6
PEEING_ALARM_ENTER_HIT_FRACTION = 0.65
PEEING_ALARM_EXIT_HIT_FRACTION = 0.45
PEEING_ALARM_MIN_SAMPLES = 13
PEEING_SQUAT_HIP_KNEE_GAP_MAX = 0.09
PEEING_SQUAT_DEPTH_SCALE = 0.11
PEEING_POSE_MODEL_PATH = str(
    Path.home() / ".cache" / "trash_detection_worker" / "pose_landmarker_lite.task"
)
PEEING_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
