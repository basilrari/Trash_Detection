# worker-python/settings.py
"""
Local video pipeline configuration.

**Single source of truth:** edit the literals in this file. Nothing is read from the OS
environment or ``.env`` for application settings.

``python worker.py`` does not override ``settings``; change ``GATE_MODE`` and stride fields in this file.
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

_w = Path(__file__).resolve().parent / "weights"

# --- Scene YOLO: ``YOLO_RUNTIME`` ``"pt"`` (Ultralytics ``.pt``) or ``"engine"`` (TensorRT) ---
YOLO_RUNTIME = "engine"
YOLO_MODEL_PATH = str(_w / "yolo11x.pt")
YOLO_ENGINE_PATH = str(_w / "yolo11x_dynamic_b8_fp16_tensorRT.engine")
YOLO_TRT_BATCH_SIZE = 8
YOLO_TRT_IMAGE_SIZE = 640
YOLO_TRT_DYNAMIC = True

# --- License-plate YOLO: ``LP_RUNTIME`` ``"pt"`` or ``"engine"`` ---
LP_RUNTIME = "engine"
LP_MODEL_PATH = str(_w / "bestlicense.pt")
LP_ENGINE_PATH = str(_w / "lp_dynamic_b16_fp16_tensorRT.engine")
LP_TRT_BATCH_SIZE = 16
LP_TRT_IMAGE_SIZE = 640
LP_TRT_DYNAMIC = True
LP_CONFIDENCE = 0.25

# Plate / OCR lock-in (``VehicleLpOcrCache`` in ``pipelines/test_pipeline``).
OCR_LOCK_CONFIDENCE = 0.90
OCR_STABLE_OBSERVATIONS = 2
OCR_REFRESH_STRIDE = 20
LP_LOCK_REFRESH_STRIDE = 10

# LabelAnnotator ``smart_position`` adds layout work; set False for faster drawing.
ANNOTATOR_SMART_POSITION = False

# --- OCR prefilter (``models/ocr.py``) ---
OCR_MIN_PLATE_SIDE = 12
OCR_MIN_VARIANCE_LAPLACIAN = 0.0  # >0 to skip very blurry crops (e.g. 30.0); 0 disables.

# Scene YOLO micro-batch size in ``GATE_MODE=off`` chunk sub-batching (see ``test_pipeline``).
YOLO_MICRO_BATCH_SIZE = 8

# --- Gating (``core/yolo_stride_gate.py`` / ``pipelines.test_pipeline``) ---
# GATE_MODE:
#   ``"off"`` — time-chunk path: YOLO on every decoded frame (sub-batched by ``YOLO_MICRO_BATCH_SIZE``).
#   ``"yolo"`` — coarse/dense stride gate (``YOLO_COARSE_STRIDE`` / ``YOLO_DENSE_STRIDE``).
#   ``"1"``, ``"2"``, … — uniform scene-YOLO stride: run YOLO only on frames where ``index % N == 0``,
#     micro-batched in windows of ``N * YOLO_MICRO_BATCH_SIZE`` reads; other frames reuse the last scene boxes.
GATE_MODE = "2"
YOLO_COARSE_STRIDE = 10
YOLO_DENSE_STRIDE = 2
YOLO_DENSE_IDLE_MISS_STREAK = 8

# --- RF-DETR trash / cigarette (TensorRT engines only) ---
TRASH_ENGINE_PATH = str(_w / "trash_fp16_tensorRT.engine")
CIGARETTE_ENGINE_PATH = str(_w / "cigarette_fp16_tensorRT.engine")
TRASH_CONFIDENCE = 0.4

# Extra ``[TRT]`` timing lines from ``models/rfdetr_trt_trash.py``.
RF_DETR_TRT_TIMING = False

# RF-DETR preprocess: use CPU (NumPy + OpenCV) unless you set a CUDA opt-in value here:
# ``"1"``, ``"true"``, ``"yes"``, ``"on"``, ``"cuda"``, ``"auto"`` (CUDA when available).
# ``""``, ``"0"``, ``"cpu"``, ``"false"``, ``"off"``, ``"no"`` → CPU.
RF_DETR_PREPROCESS_CUDA = "1"

# Run the cigarette TRT head on 1/N RF-DETR batches only (1 = every batch). Saves ~half TRT when N=2 if heads dominate.
# Benchmark wall time with RF-DETR TRT batch metrics in the pipeline summary; pair with rebuilding engines (larger batch) or YOLO/LP TensorRT separately if needed.
RF_DETR_CIGARETTE_EVERY_N_BATCHES = 1

# Max frames the oldest RF-DETR-queued real frame may wait before a padded tail flush (0 = only flush at full batch or EOF).
RF_DETR_MAX_QUEUE_LATENCY_FRAMES = 0

# --- PaddleOCR (``models/ocr.py``) ---
# ``""`` → auto (GPU if Paddle sees CUDA). Otherwise ``"cpu"``, ``"gpu"``, ``"gpu:0"``, etc.
PADDLE_OCR_DEVICE = "gpu"
# ``None`` → default isolation rule (Blackwell + GPU OCR). ``True`` / ``False`` to force.
PADDLE_OCR_ISOLATE_PROCESS: bool | None = None

# Re-run LP (+ downstream OCR) for a vehicle at most every N frames when the same vehicle is tracked (IoU match). 1 = every frame.
LP_VEHICLE_LP_STRIDE = 3

# Decode thread queue depth (0 = read frames synchronously in the main loop).
PIPELINE_READ_AHEAD_QUEUE_SIZE = 8
# Async video writer queue depth (0 = call ``write`` on the main thread).
PIPELINE_WRITE_QUEUE_SIZE = 8

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
