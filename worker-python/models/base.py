# worker-python/models/base.py
from abc import ABC, abstractmethod
from typing import List, Tuple
from core.types import FrameData, Detection, LicensePlate, PoseResult

class Detector(ABC):
    @abstractmethod
    def detect(self, frames: List[FrameData]) -> List[List[Detection]]:
        pass

class LicensePlateDetector(ABC):
    @abstractmethod
    def detect_plates(self, frames: List[FrameData]) -> List[List[LicensePlate]]:
        pass

class OCRModel(ABC):
    @abstractmethod
    def recognize(self, crops: List[Tuple[float, float, float, float]]) -> List[str]:
        pass

class PoseEstimator(ABC):
    @abstractmethod
    def estimate(self, frames: List[FrameData]) -> List[List[PoseResult]]:
        pass

class TrashDetector(ABC):
    @abstractmethod
    def detect_trash(self, frames: List[FrameData]) -> List[List[Detection]]:
        pass