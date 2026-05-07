# worker-python/models/ocr.py
import os
from typing import List, Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR

from models.base import OCRModel


class Ocr(OCRModel):
    """
    PaddleOCR wrapper returning (text, confidence) per crop (OpenCV BGR).
    Device is controlled by ``PADDLE_OCR_DEVICE`` (default ``cpu``). Use ``gpu``
    only with ``paddlepaddle-gpu`` installed.
    """

    def __init__(self, lang: str = "en"):
        device = os.environ.get("PADDLE_OCR_DEVICE", "cpu")
        self._ocr = PaddleOCR(
            lang=lang,
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )

    def _prep(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            raise ValueError("None image passed to OCR")
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype("uint8")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _run_ocr(self, img: np.ndarray):
        ocr = self._ocr
        if hasattr(ocr, "predict"):
            return ocr.predict(img)
        try:
            return ocr.ocr(img, cls=False)
        except TypeError:
            return ocr.ocr(img)

    def recognize(self, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        outputs: List[Tuple[str, float]] = []

        for crop in crops:
            try:
                if crop is None or crop.size == 0:
                    outputs.append(("", 0.0))
                    continue

                img = self._prep(crop)
                res = self._run_ocr(img)

                if not res:
                    outputs.append(("", 0.0))
                    continue

                if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
                    page_res = res[0]
                    rec_texts = page_res.get("rec_texts", [])
                    rec_scores = page_res.get("rec_scores", [])
                    if not rec_texts:
                        outputs.append(("", 0.0))
                        continue
                    dets = list(zip(rec_texts, rec_scores))
                elif isinstance(res, list) and all(
                    isinstance(d, (list, tuple)) and len(d) == 2 for d in res
                ):
                    dets = [d[1] for d in res]
                else:
                    outputs.append(("", 0.0))
                    continue

                best_text = ""
                best_score = 0.0
                for txt_score in dets:
                    try:
                        if isinstance(txt_score, (list, tuple)) and len(txt_score) >= 2:
                            txt = str(txt_score[0]).strip()
                            score = float(txt_score[1])
                        else:
                            txt = str(txt_score).strip()
                            score = 0.0
                    except (ValueError, TypeError):
                        continue

                    if txt and score >= best_score:
                        best_text = txt
                        best_score = score
                    elif txt and best_score == 0.0 and not best_text:
                        best_text = txt

                outputs.append((best_text, best_score))

            except Exception:
                outputs.append(("", 0.0))

        return outputs
