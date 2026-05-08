# worker-python/models/ocr.py
import atexit
import logging
import multiprocessing as mp
import os
import queue
import threading
from typing import Any, List, Tuple

import cv2
import numpy as np

from models.base import OCRModel

logger = logging.getLogger(__name__)


def _resolve_paddle_ocr_device() -> str:
    """Pick Paddle device: env ``PADDLE_OCR_DEVICE`` if set, else ``gpu`` when CUDA is available."""
    raw = os.environ.get("PADDLE_OCR_DEVICE", "").strip()
    if raw:
        return raw
    try:
        import paddle

        if paddle.device.cuda.device_count() > 0:
            return "gpu"
    except Exception:
        pass
    return "cpu"


def _is_blackwell_visible() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        cap = torch.cuda.get_device_capability(0)
        return int(cap[0]) >= 12
    except Exception:
        return False


def _isolate_ocr_process_default(device: str) -> bool:
    """Default isolation policy.

    On Blackwell + GPU OCR we isolate PaddleOCR in a subprocess by default because
    mixed PyTorch + Paddle + TensorRT in one process can poison CUDA context state.
    """
    v = os.environ.get("PADDLE_OCR_ISOLATE_PROCESS", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return device.startswith("gpu") and _is_blackwell_visible()


def _prep_image(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("None image passed to OCR")
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype("uint8")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _run_ocr_backend(ocr: Any, img: np.ndarray):
    if hasattr(ocr, "predict"):
        return ocr.predict(img)
    try:
        return ocr.ocr(img, cls=False)
    except TypeError:
        return ocr.ocr(img)


def _pick_best_text_score(res: Any) -> tuple[str, float]:
    if not res:
        return ("", 0.0)

    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
        page_res = res[0]
        rec_texts = page_res.get("rec_texts", [])
        rec_scores = page_res.get("rec_scores", [])
        if not rec_texts:
            return ("", 0.0)
        dets = list(zip(rec_texts, rec_scores))
    elif isinstance(res, list) and all(isinstance(d, (list, tuple)) and len(d) == 2 for d in res):
        dets = [d[1] for d in res]
    else:
        return ("", 0.0)

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
    return (best_text, best_score)


def _recognize_with_backend(ocr: Any, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
    outputs: List[Tuple[str, float]] = []
    for crop in crops:
        try:
            if crop is None or crop.size == 0:
                outputs.append(("", 0.0))
                continue
            img = _prep_image(crop)
            res = _run_ocr_backend(ocr, img)
            outputs.append(_pick_best_text_score(res))
        except Exception:
            outputs.append(("", 0.0))
    return outputs


def _ocr_worker_main(lang: str, device: str, req_q: Any, resp_q: Any) -> None:
    try:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(
            lang=lang,
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        resp_q.put(("__ready__", None, None))
    except Exception as exc:
        resp_q.put(("__fatal__", None, f"{type(exc).__name__}: {exc}"))
        return

    while True:
        item = req_q.get()
        if item is None:
            break
        req_id, crops = item
        try:
            out = _recognize_with_backend(ocr, crops)
            resp_q.put((req_id, out, None))
        except Exception as exc:
            resp_q.put((req_id, [("", 0.0)] * len(crops), f"{type(exc).__name__}: {exc}"))


class Ocr(OCRModel):
    """
    PaddleOCR wrapper returning (text, confidence) per crop (OpenCV BGR).

    Device: ``PADDLE_OCR_DEVICE`` overrides (e.g. ``cpu``, ``gpu``, ``gpu:0``). If unset,
    uses **GPU** when ``paddlepaddle-gpu`` sees at least one CUDA device, otherwise **CPU**.

    On Blackwell + GPU OCR, process isolation is enabled by default so Paddle runs in a
    child process (``spawn``) and does not share CUDA context state with torch/TRT.
    Override with ``PADDLE_OCR_ISOLATE_PROCESS=0|1``.
    """

    def __init__(self, lang: str = "en"):
        self._device = _resolve_paddle_ocr_device()
        self._isolate = _isolate_ocr_process_default(self._device)
        self._ocr: Any | None = None
        self._proc: mp.Process | None = None
        self._req_q: Any | None = None
        self._resp_q: Any | None = None
        self._ipc_lock = threading.Lock()
        self._next_req_id = 1

        if self._isolate:
            self._start_worker(lang=lang, device=self._device)
        else:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                lang=lang,
                device=self._device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
        logger.info("PaddleOCR device=%s isolate_process=%s", self._device, self._isolate)
        atexit.register(self.close)

    def _start_worker(self, *, lang: str, device: str) -> None:
        ctx = mp.get_context("spawn")
        req_q = ctx.Queue(maxsize=2)
        resp_q = ctx.Queue(maxsize=2)
        proc = ctx.Process(
            target=_ocr_worker_main,
            args=(lang, device, req_q, resp_q),
            daemon=False,
        )
        proc.start()
        try:
            tag, _, err = resp_q.get(timeout=180.0)
        except queue.Empty as exc:
            proc.terminate()
            proc.join(timeout=2.0)
            raise RuntimeError("PaddleOCR worker did not start in time.") from exc
        if tag == "__fatal__":
            proc.join(timeout=2.0)
            raise RuntimeError(f"PaddleOCR worker failed to initialize: {err}")
        if tag != "__ready__":
            proc.terminate()
            proc.join(timeout=2.0)
            raise RuntimeError(f"Unexpected PaddleOCR worker handshake: {tag!r}")
        self._proc = proc
        self._req_q = req_q
        self._resp_q = resp_q

    @property
    def paddle_device(self) -> str:
        """Resolved Paddle runtime (``gpu``, ``cpu``, or explicit env value)."""
        return self._device

    @property
    def isolate_process(self) -> bool:
        """Whether OCR is running in a separate process."""
        return self._isolate

    def close(self) -> None:
        proc = self._proc
        req_q = self._req_q
        if proc is None:
            return
        try:
            if req_q is not None:
                req_q.put_nowait(None)
        except Exception:
            pass
        try:
            proc.join(timeout=10.0)
        except Exception:
            pass
        if proc.is_alive():
            try:
                proc.terminate()
                proc.join(timeout=5.0)
            except Exception:
                pass
        self._proc = None
        self._req_q = None
        self._resp_q = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def recognize(self, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        if not crops:
            return []
        if not self._isolate:
            assert self._ocr is not None
            return _recognize_with_backend(self._ocr, crops)

        proc = self._proc
        req_q = self._req_q
        resp_q = self._resp_q
        if proc is None or req_q is None or resp_q is None or not proc.is_alive():
            logger.error("PaddleOCR worker process is not alive; returning empty OCR outputs.")
            return [("", 0.0)] * len(crops)

        with self._ipc_lock:
            req_id = self._next_req_id
            self._next_req_id += 1
            req_q.put((req_id, crops))
            try:
                rid, out, err = resp_q.get(timeout=180.0)
            except queue.Empty:
                logger.error("PaddleOCR worker timed out; returning empty OCR outputs.")
                return [("", 0.0)] * len(crops)

        if rid != req_id:
            logger.error("PaddleOCR worker response id mismatch (%r vs %r).", rid, req_id)
            return [("", 0.0)] * len(crops)
        if err:
            logger.error("PaddleOCR worker error: %s", err)
            return [("", 0.0)] * len(crops)
        return out if isinstance(out, list) else [("", 0.0)] * len(crops)
