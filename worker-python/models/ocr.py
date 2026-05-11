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
    """Paddle device from ``settings.PADDLE_OCR_DEVICE`` (required; no implicit CPU fallback)."""
    from settings import PADDLE_OCR_DEVICE

    raw = str(PADDLE_OCR_DEVICE).strip()
    if not raw:
        raise ValueError("PADDLE_OCR_DEVICE must be set explicitly (e.g. 'gpu' or 'cpu').")
    low = raw.lower()
    if low.startswith("gpu") or low.startswith("cuda"):
        try:
            import paddle
        except ImportError as exc:
            raise RuntimeError(
                "PADDLE_OCR_DEVICE requests GPU/CUDA but paddle is not importable."
            ) from exc
        try:
            if int(paddle.device.cuda.device_count()) <= 0:
                raise RuntimeError(
                    f"PADDLE_OCR_DEVICE is {raw!r} but paddle.device.cuda.device_count() is 0."
                )
        except Exception as exc:
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(
                f"Could not verify Paddle CUDA devices for PADDLE_OCR_DEVICE={raw!r}."
            ) from exc
    return raw


def _isolate_ocr_process_from_settings() -> bool:
    """Isolation flag from ``settings.PADDLE_OCR_ISOLATE_PROCESS`` (must be a bool)."""
    from settings import PADDLE_OCR_ISOLATE_PROCESS

    if not isinstance(PADDLE_OCR_ISOLATE_PROCESS, bool):
        raise TypeError("PADDLE_OCR_ISOLATE_PROCESS must be True or False (no implicit heuristic).")
    return PADDLE_OCR_ISOLATE_PROCESS


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


def _ocr_crop_too_small_or_blurry(bgr: np.ndarray) -> bool:
    """True if crop should skip Paddle (saves time on tiny / very soft patches)."""
    try:
        from settings import OCR_MIN_PLATE_SIDE, OCR_MIN_VARIANCE_LAPLACIAN
    except Exception:
        return False
    if bgr is None or bgr.size == 0:
        return True
    h, w = bgr.shape[:2]
    if min(h, w) < int(OCR_MIN_PLATE_SIDE):
        return True
    thr = float(OCR_MIN_VARIANCE_LAPLACIAN)
    if thr <= 0:
        return False
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    v = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return v < thr


def _pick_best_text_score(res: Any) -> tuple[str, float]:
    if not res:
        return ("", 0.0)

    # PaddleOCR ``predict`` on a batch returns one dict per image; single-image path may return ``[dict]``.
    if isinstance(res, dict):
        res = [res]

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
    """Run PaddleOCR on all valid crops in one ``predict([...])`` call (no silent per-crop fallback)."""
    n = len(crops)
    outputs: List[Tuple[str, float]] = [("", 0.0)] * n
    valid_idx: List[int] = []
    images: List[np.ndarray] = []
    for i, crop in enumerate(crops):
        if crop is None or crop.size == 0:
            continue
        if _ocr_crop_too_small_or_blurry(crop):
            continue
        try:
            images.append(_prep_image(crop))
            valid_idx.append(i)
        except Exception as exc:
            raise RuntimeError(f"OCR crop preprocess failed at index {i}.") from exc
    if not images:
        return outputs

    if not hasattr(ocr, "predict"):
        raise RuntimeError(
            "PaddleOCR instance has no predict(); upgrade paddleocr or adjust the integration."
        )

    raw = ocr.predict(images)
    if not isinstance(raw, list):
        raw = [raw]
    if len(raw) != len(images):
        raise RuntimeError(
            f"PaddleOCR predict returned {len(raw)} result(s) for {len(images)} image(s) (expected 1:1)."
        )
    for slot, idx in enumerate(valid_idx):
        outputs[idx] = _pick_best_text_score(raw[slot])
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

    Device: ``PADDLE_OCR_DEVICE`` (required), e.g. ``gpu`` or ``cpu``. GPU requests fail if Paddle
    reports no CUDA devices.

    ``PADDLE_OCR_ISOLATE_PROCESS`` must be explicit ``True`` or ``False`` (no heuristic).
    """

    def __init__(self, lang: str = "en"):
        self._device = _resolve_paddle_ocr_device()
        self._isolate = _isolate_ocr_process_from_settings()
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
            raise RuntimeError("PaddleOCR worker process is not alive.")

        with self._ipc_lock:
            req_id = self._next_req_id
            self._next_req_id += 1
            req_q.put((req_id, crops))
            try:
                rid, out, err = resp_q.get(timeout=180.0)
            except queue.Empty as exc:
                raise RuntimeError("PaddleOCR worker timed out.") from exc

        if rid != req_id:
            raise RuntimeError(f"PaddleOCR worker response id mismatch ({rid!r} vs {req_id!r}).")
        if err:
            raise RuntimeError(f"PaddleOCR worker error: {err}")
        if not isinstance(out, list) or len(out) != len(crops):
            raise RuntimeError(
                f"PaddleOCR worker returned malformed output (expected {len(crops)} items)."
            )
        return out
