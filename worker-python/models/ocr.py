# worker-python/models/ocr.py
from typing import List, Tuple
import cv2
import numpy as np
from paddleocr import PaddleOCR
from models.base import OCRModel

class Ocr(OCRModel):
    """
    Minimal PaddleOCR wrapper that runs on GPU and returns (text, confidence)
    for each input crop. Input crops are OpenCV images (BGR numpy arrays).
    """

    def __init__(self, lang: str = "en"):
        # Force GPU usage; if Paddle/GPU not available this will raise.
        self._ocr = PaddleOCR(lang=lang, device="gpu")

    def _prep(self, img: np.ndarray) -> np.ndarray:
        # Ensure uint8 and convert BGR->RGB
        if img is None:
            raise ValueError("None image passed to OCR")
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype("uint8")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def recognize(self, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        :param crops: list of numpy arrays (OpenCV BGR)
        :return: list of tuples (text, confidence). If nothing found, returns ("", 0.0)
        """
        outputs: List[Tuple[str, float]] = []

        for idx, crop in enumerate(crops):
            try:
                if crop is None or crop.size == 0:
                    outputs.append(("", 0.0))
                    continue

                img = self._prep(crop)

                # Call PaddleOCR API
                try:
                    res = self._ocr.ocr(img, cls=False)
                except TypeError:
                    res = self._ocr.ocr(img)

                # Normalize result: handle both old [ [box, (text, score)], ... ] and new dict formats
                if not res:
                    outputs.append(("", 0.0))
                    continue

                # Check for new dict format (as in your debug)
                if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
                    page_res = res[0]
                    rec_texts = page_res.get('rec_texts', [])
                    rec_scores = page_res.get('rec_scores', [])
                    if not rec_texts:
                        outputs.append(("", 0.0))
                        continue
                    dets = list(zip(rec_texts, rec_scores))
                # Old format: list of [box, (text, score)]
                elif isinstance(res, list) and all(isinstance(d, (list, tuple)) and len(d) == 2 for d in res):
                    dets = [d[1] for d in res]  # extract (text, score) tuples
                else:
                    outputs.append(("", 0.0))
                    continue

                # Choose the detection with highest score
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

            except Exception as e:
                outputs.append(("", 0.0))

        return outputs
        """
        :param crops: list of numpy arrays (OpenCV BGR)
        :return: list of tuples (text, confidence). If nothing found, returns ("", 0.0)
        """
        outputs: List[Tuple[str, float]] = []

        # Open debug file in append mode
        with open("ocr_debug.txt", "a") as debug_file:
            for idx, crop in enumerate(crops):
                debug_file.write(f"\n=== Crop {idx} ===\n")
                try:
                    if crop is None or crop.size == 0:
                        debug_file.write("Invalid crop: None or empty\n")
                        outputs.append(("", 0.0))
                        continue

                    debug_file.write(f"Crop shape: {crop.shape}, dtype: {crop.dtype}\n")

                    img = self._prep(crop)

                    # Call PaddleOCR API
                    try:
                        res = self._ocr.ocr(img, cls=False)
                    except TypeError:
                        res = self._ocr.ocr(img)

                    # Dump raw res to file (JSON for readability)
                    debug_file.write("Raw PaddleOCR res:\n")
                    try:
                        debug_file.write(json.dumps(res, indent=2, default=str) + "\n")
                    except Exception as dump_e:
                        debug_file.write(f"Failed to dump res: {dump_e}\n{str(res)}\n")

                    # Normalize result: handle both old [ [box, (text, score)], ... ] and new dict formats
                    if not res:
                        debug_file.write("No valid res\n")
                        outputs.append(("", 0.0))
                        continue

                    # Check for new dict format (as in your debug)
                    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
                        page_res = res[0]
                        rec_texts = page_res.get('rec_texts', [])
                        rec_scores = page_res.get('rec_scores', [])
                        if not rec_texts:
                            debug_file.write("Empty rec_texts\n")
                            outputs.append(("", 0.0))
                            continue
                        dets = list(zip(rec_texts, rec_scores))
                    # Old format: list of [box, (text, score)]
                    elif isinstance(res, list) and all(isinstance(d, (list, tuple)) and len(d) == 2 for d in res):
                        dets = [d[1] for d in res]  # extract (text, score) tuples
                    else:
                        debug_file.write(f"Unknown res format: {type(res)}\n")
                        outputs.append(("", 0.0))
                        continue

                    # Choose the detection with highest score
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
                            debug_file.write(f"Malformed txt_score: {txt_score}\n")
                            continue

                        if txt and score >= best_score:
                            best_text = txt
                            best_score = score
                        elif txt and best_score == 0.0 and not best_text:
                            best_text = txt

                    outputs.append((best_text, best_score))
                    debug_file.write(f"Parsed output: ({best_text}, {best_score})\n")

                except Exception as e:
                    debug_file.write(f"Error in crop {idx}: {str(e)}\n{traceback.format_exc()}\n")
                    outputs.append(("", 0.0))

        return outputs