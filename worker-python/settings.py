# worker-python/settings.py
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
REDIS_URL = os.getenv('REDIS_URL')

# Add your chunk sizes, thresholds, model paths here
# Config for chunking and thresholds (adjust as needed)
CHUNK_SECONDS = 5
YOLO_CONFIDENCE = 0.5
PLATE_CONFIDENCE = 0.5

LP_MODEL_PATH = 'path/to/lp_model.pt'
RFDETR_MODEL_PATH = 'path/to/rfdetr.pt'