# worker-python/settings.py
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Default input/output when no CLI args are passed (edit here or use: python worker.py video.mp4 -o out.mp4)
VIDEO_PATH = os.getenv("VIDEO_PATH", "Test.mp4")
OUTPUT_VIDEO = os.getenv("OUTPUT_VIDEO", "output_with_boxes.mp4")

CHUNK_SECONDS = 5
YOLO_CONFIDENCE = 0.5
PLATE_CONFIDENCE = 0.5

LP_MODEL_PATH = "path/to/lp_model.pt"
RFDETR_MODEL_PATH = "path/to/rfdetr.pt"
