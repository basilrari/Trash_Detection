# portable_peeing/settings.py — peeing + pose + video encode (no scene detector).
"""
Edit literals here. Scene person boxes come from **your** detector (e.g. DFINE); see ``DFINE_INTEGRATION.md``.
"""
from pathlib import Path

INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"

VIDEO_PATH = f"{INPUTS_DIR}/clip.mp4"
OUTPUT_VIDEO = f"{OUTPUTS_DIR}/annotated.mp4"

INPUT_VIDEO_FPS_MIN = 5
INPUT_VIDEO_FPS_MAX = 60

OUTPUT_VIDEO_ENCODER = "auto"
FFMPEG_PATH = "ffmpeg"
NVENC_PRESET = "p1"
NVENC_CQ = 28

# Minimum score for ``person`` / motorcycle rows passed into PeeingDetector.update (``yolo_conf`` arg).
PEEING_DETECTION_CONFIDENCE = 0.4

_w = Path(__file__).resolve().parent / "weights"

# Decode-window size hint when batching external detector calls (was YOLO_MICRO_BATCH_SIZE).
DETECTION_WINDOW_BATCH = 8

# ~5 D-FINE samples per second of video (stride = round(fps / this), clamped in frame_stride).
SCENE_YOLO_TARGET_FRAMES_PER_SECOND = 5
FRAME_SAMPLE_STRIDE_OVERRIDE: int | None = None

PIPELINE_READ_AHEAD_QUEUE_SIZE = 8
PIPELINE_WRITE_QUEUE_SIZE = 8

PEEING_CROP_MARGIN = 0.12
PEEING_MIN_VISIBILITY = 0.45
PEEING_HAND_GROIN_Y_THRESHOLD = 0.09
PEEING_SECONDS_REQUIRED = 10
PEEING_MIN_HITS_PER_SECOND = 3
PEEING_TRACK_IOU_THRESHOLD = 0.35
PEEING_TRACK_MAX_MISSED_SECONDS = 3.0
PEEING_STILL_SECONDS_REQUIRED = 1.0
PEEING_STILL_MAX_CENTER_MOTION = 0.08
PEEING_STILL_MAX_SIZE_CHANGE = 0.20
PEEING_STILL_MIN_IOU = 0.65

PEEING_MOTORCYCLE_EXCLUSION_ENABLED = True
PEEING_MOTORCYCLE_LABELS = ("motorcycle", "motorbike")
PEEING_MOTORCYCLE_BBOX_EXPAND_X = 0.15
PEEING_MOTORCYCLE_BBOX_EXPAND_Y = 0.10
PEEING_MOTORCYCLE_LOWER_BODY_FRACTION = 0.60
PEEING_MOTORCYCLE_LOWER_OVERLAP_THRESHOLD = 0.10

# Default for standalone portable_peeing runs; Basil_Test/draw_dfine_peeing.py overrides to repo .pt.
PEEING_YOLO_POSE_MODEL = str(_w / "yolo11n-pose.pt")
PEEING_YOLO_POSE_BATCH_SIZE = 8
PEEING_YOLO_POSE_TRT_DYNAMIC = True
PEEING_YOLO_POSE_IMGSZ = 640
PEEING_YOLO_POSE_DEVICE: str | None = None
PEEING_YOLO_POSE_TRT_TIMING = False
PEEING_YOLO_POSE_CROSS_FRAME_BATCH = True
PEEING_YOLO_POSE_PREFETCH_DEBUG = False

PEEING_MAX_POSE_PERSONS_PER_FRAME: int | None = None
PEEING_PERSIST_POSE_VIZ = True
PEEING_DEBUG_TIMING = True
