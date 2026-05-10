# worker-python/models/types.py
from typing import NamedTuple, List, Tuple, Optional


class FrameData(NamedTuple):
    index: int
    timestamp: float
    image: object  # e.g., np.ndarray or reference


class Detection(NamedTuple):
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    label: str
    confidence: float


class LicensePlate(NamedTuple):
    bbox: Tuple[float, float, float, float]
    text: str
    confidence: float


class PoseResult(NamedTuple):
    keypoints: List[Tuple[float, float, float]]  # x, y, visibility for each landmark
    confidence: float


class Event(NamedTuple):
    timestamp: float
    event_type: str  # 'vehicle', 'urination', 'litter', etc.
    bbox: Optional[Tuple[float, float, float, float]]
    confidence: float
    extra: Optional[str]  # e.g., plate text
