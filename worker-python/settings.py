# worker-python/settings.py
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
REDIS_URL = os.getenv('REDIS_URL')

# Add your chunk sizes, thresholds, model paths here
CHUNK_SECONDS = 60  # e.g., 60-second chunks
YOLO_CONFIDENCE = 0.5
LP_MODEL_PATH = 'path/to/lp_model.pt'
RFDETR_MODEL_PATH = 'path/to/rfdetr.pt'