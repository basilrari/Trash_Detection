# worker-python/models/ocr.py

from typing import List

import numpy as np
from paddleocr import PaddleOCR

from models.base import OCRModel


class Ocr(OCRModel):
    """
    OCR wrapper around PaddleOCR that matches the OCRModel interface.
    """

    def __init__(self, lang: str = "en"):
        # Use GPU if available (set device="cpu" to force CPU)
        self._ocr = PaddleOCR(device="gpu", lang=lang)

    def recognize(self, crops: List[np.ndarray]) -> List[str]:
        """
        Recognize text from a list of image crops.

        :param crops: List of BGR or RGB image arrays (OpenCV style).
        :return: List of recognized text strings (or 'unknown' if none).
        """
        texts: List[str] = []

        for crop in crops:
            # Try the legacy API first (many versions accept cls argument)
            try:
                result = self._ocr.ocr(crop, cls=False)
            except TypeError:
                # Fallback 1: new API might accept ocr(crop) without cls
                try:
                    result = self._ocr.ocr(crop)
                except TypeError:
                    # Fallback 2: some PaddleOCR releases expose predict()
                    try:
                        result = self._ocr.predict(crop)
                    except Exception as e:
                        # All attempts failed — raise a clear error
                        raise RuntimeError(
                            "Failed to call PaddleOCR. Tried `.ocr(crop, cls=False)`, "
                            "`.ocr(crop)`, and `.predict(crop)` but all failed."
                        ) from e

            # Normalize result shape: many API variants return a list-of-lists
            # Example result for one image: [ [ [box, (text, score)], ... ] ]
            if not result:
                texts.append("unknown")
                continue

            # If result is nested (list with first element the detections), normalize it
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                detections = result[0]
            else:
                detections = result

            if not detections:
                texts.append("unknown")
                continue

            # Each detection is typically [box, (text, score)] or similar.
            # Guard against unexpected shapes.
            try:
                best_text = detections[0][1][0]
                if not best_text:
                    raise ValueError
            except Exception:
                # Fallback: try to stringify whatever structure we have
                try:
                    # Attempt to find the first text-like element in detections
                    found = None
                    for det in detections:
                        if isinstance(det, (list, tuple)) and len(det) >= 2:
                            cand = det[1]
                            if isinstance(cand, (list, tuple)) and len(cand) >= 1:
                                if isinstance(cand[0], str):
                                    found = cand[0]
                                    break
                    if found:
                        texts.append(found)
                        continue
                except Exception:
                    pass

                texts.append("unknown")
                continue

            texts.append(best_text)

        return texts
 