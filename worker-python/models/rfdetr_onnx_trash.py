"""Optional RF-DETR trash heads via ONNX Runtime (exported ``inference_model.onnx``).

Post-processing matches ``rfdetr`` export benchmark (top-300, cxcywh→xyxy, scale to
original frame size). See ``scripts/export_rfdetr_heads.py`` and Roboflow export docs.

Pipeline: set ``RF_DETR_BACKEND=onnx`` and ``RF_DETR_TRASH_ONNX`` / ``RF_DETR_CIGARETTE_ONNX``.
Install: ``pip install onnxruntime-gpu`` (or ``onnxruntime`` for CPU).
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import cv2
import numpy as np
import supervision as sv
import torch

from core.types import Detection, FrameData
from models.base import TrashDetector
from models.trash_detector import _parallel_rfdetr_heads_enabled, _sv_to_detections

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def _box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        (x_c - 0.5 * w.clamp(min=0.0)),
        (y_c - 0.5 * h.clamp(min=0.0)),
        (x_c + 0.5 * w.clamp(min=0.0)),
        (y_c + 0.5 * h.clamp(min=0.0)),
    ]
    return torch.stack(b, dim=-1)


def _post_process(
    out_bbox: torch.Tensor,
    out_logits: torch.Tensor,
    target_sizes: torch.Tensor,
    *,
    topk: int = 300,
) -> list[dict[str, torch.Tensor]]:
    """Same logic as ``rfdetr.export.benchmark.post_process`` (per batch row)."""
    assert out_logits.shape[0] == target_sizes.shape[0]
    assert target_sizes.shape[1] == 2
    prob = out_logits.sigmoid()
    bsz = out_logits.shape[0]
    flat = prob.view(bsz, -1)
    k = min(topk, flat.shape[1])
    topk_values, topk_indexes = torch.topk(flat, k, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = _box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    return [
        {"scores": score, "labels": label, "boxes": box}
        for score, label, box in zip(scores, labels, boxes)
    ]


def _preprocess_rgb_list(
    images_rgb: List[np.ndarray],
    out_h: int,
    out_w: int,
) -> tuple[np.ndarray, torch.Tensor]:
    """Resize + ImageNet normalize → ``NCHW`` float32 batch; ``target_sizes`` ``[B,2]`` = (H,W) original."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
    rows: list[np.ndarray] = []
    sizes: list[tuple[int, int]] = []
    for img in images_rgb:
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Expected RGB HWC uint8/float images")
        h0, w0 = int(img.shape[0]), int(img.shape[1])
        sizes.append((h0, w0))
        if img.dtype != np.uint8:
            im = np.clip(img, 0, 255).astype(np.uint8)
        else:
            im = img
        r = cv2.resize(im, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        x = r.astype(np.float32) / 255.0
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))
        rows.append(x)
    batch = np.stack(rows, axis=0).astype(np.float32)
    ts = torch.tensor(sizes, dtype=torch.float32, device="cpu")
    return batch, ts


def _ort_providers() -> list[Any]:
    """``RF_DETR_ONNX_PROVIDERS``: ``cuda`` (default), ``cpu``, or ``tensorrt`` (TensorRT EP)."""
    raw = os.environ.get("RF_DETR_ONNX_PROVIDERS", "cuda").strip().lower()
    if raw == "cpu":
        return ["CPUExecutionProvider"]
    if raw in ("trt", "tensorrt"):
        return [
            (
                "TensorrtExecutionProvider",
                {"device_id": int(os.environ.get("RF_DETR_TRT_DEVICE_ID", "0"))},
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


class _OnnxRfDetrHead:
    def __init__(
        self,
        onnx_path: Path,
        *,
        default_label: str,
        conf_threshold: float,
        class_names: dict[int, str] | None,
        use_sv_names: bool,
    ) -> None:
        if ort is None:
            raise ImportError("onnxruntime is required for RF_DETR_BACKEND=onnx (pip install onnxruntime-gpu)")
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        self._onnx_path = onnx_path
        self._default_label = default_label
        self._conf = float(conf_threshold)
        self._class_names = dict(class_names) if class_names else None
        self._use_sv_names = use_sv_names
        self._sess = ort.InferenceSession(str(onnx_path), providers=_ort_providers())
        inp0 = self._sess.get_inputs()[0]
        self._input_name = inp0.name
        shape = inp0.shape
        if len(shape) != 4:
            raise ValueError(f"Expected NCHW ONNX input, got shape {shape}")
        self._batch = int(shape[0]) if isinstance(shape[0], int) and shape[0] > 0 else -1
        self._in_c = int(shape[1])
        self._in_h = int(shape[2])
        self._in_w = int(shape[3])
        if self._batch < 1:
            raise ValueError(
                "Exported ONNX must use a **static** positive batch dimension for this backend "
                f"(got batch dim {shape[0]}). Re-export with e.g. batch_size=8."
            )

    def predict_batch(self, images_rgb: List[np.ndarray]) -> list[Any]:
        """Returns one ``sv.Detections`` per input frame (after threshold)."""
        n = len(images_rgb)
        if n == 0:
            return []
        bsz = self._batch
        out: list[Any] = []
        for start in range(0, n, bsz):
            chunk = images_rgb[start : start + bsz]
            pad = bsz - len(chunk)
            if pad > 0:
                chunk = chunk + [np.zeros_like(chunk[0])] * pad
            batch_np, target_sizes = _preprocess_rgb_list(chunk, self._in_h, self._in_w)
            t0 = time.perf_counter()
            raw = self._sess.run(None, {self._input_name: batch_np})
            if os.environ.get("RF_DETR_PROFILE", "").strip() in ("1", "true", "yes"):
                ms = (time.perf_counter() - t0) * 1000.0
                logger.info("ORT %s batch=%s %.1f ms", self._onnx_path.name, bsz, ms)

            dets_t, lbls_t = self._pick_outputs(raw)
            dets_t = torch.from_numpy(dets_t)
            lbls_t = torch.from_numpy(lbls_t)
            pp = _post_process(dets_t, lbls_t, target_sizes)
            valid = n - start
            for j in range(bsz):
                if j >= valid:
                    break
                r = pp[j]
                scores = r["scores"].detach().cpu().numpy()
                boxes = r["boxes"].detach().cpu().numpy()
                cls_ids = r["labels"].detach().cpu().numpy().astype(np.int64)
                m = scores >= self._conf
                if not np.any(m):
                    out.append(
                        sv.Detections(
                            xyxy=np.zeros((0, 4), dtype=np.float32),
                            confidence=np.zeros((0,), dtype=np.float32),
                            class_id=np.zeros((0,), dtype=np.int64),
                        )
                    )
                    continue
                xyxy = boxes[m]
                conf = scores[m]
                cid = cls_ids[m]
                sv_det = sv.Detections(
                    xyxy=xyxy.astype(np.float32),
                    confidence=conf.astype(np.float32),
                    class_id=cid,
                )
                out.append(sv_det)
        return out

    def _pick_outputs(self, raw: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Identify (dets, logits) from ORT outputs by trailing dimension."""
        if len(raw) < 2:
            raise RuntimeError(f"Expected >=2 ONNX outputs, got {len(raw)}")
        a, b = raw[0], raw[1]
        if a.shape[-1] == 4 and len(b.shape) >= 2 and b.shape[-1] != 4:
            return a, b
        if b.shape[-1] == 4 and len(a.shape) >= 2 and a.shape[-1] != 4:
            return b, a
        # Fall back: assume benchmark order (dets, labels)
        return raw[0], raw[1]


class RfDetrOnnxTrashDetector(TrashDetector):
    """Two ONNX RF-DETR heads (trash + cigarette), same ``detect_trash`` contract as PyTorch."""

    def __init__(
        self,
        trash_onnx: str | Path,
        cigarette_onnx: str | Path,
        *,
        class_names: dict[int, str] | None = None,
        conf_threshold: float = 0.4,
    ) -> None:
        self._conf = float(conf_threshold)
        self._class_names = dict(class_names) if class_names else None
        use_sv = class_names is not None
        self._heads: List[Tuple[_OnnxRfDetrHead, str]] = [
            (
                _OnnxRfDetrHead(
                    Path(trash_onnx),
                    default_label="trash",
                    conf_threshold=conf_threshold,
                    class_names=class_names,
                    use_sv_names=use_sv,
                ),
                "trash",
            ),
            (
                _OnnxRfDetrHead(
                    Path(cigarette_onnx),
                    default_label="cigarette",
                    conf_threshold=conf_threshold,
                    class_names=class_names,
                    use_sv_names=use_sv,
                ),
                "cigarette",
            ),
        ]

    def detect_trash(self, frames: List[FrameData]) -> List[List[Detection]]:
        if not frames:
            return []
        images_rgb = [cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB) for f in frames]
        merged: List[List[Detection]] = [[] for _ in frames]
        use_sv_names = self._class_names is not None
        if _parallel_rfdetr_heads_enabled() and len(self._heads) > 1:
            parts: list[Any | None] = [None] * len(self._heads)
            with ThreadPoolExecutor(max_workers=len(self._heads)) as ex:
                futures = {
                    ex.submit(self._heads[hi][0].predict_batch, images_rgb): hi
                    for hi in range(len(self._heads))
                }
                for fut in as_completed(futures):
                    hi = futures[fut]
                    parts[hi] = fut.result()
            for hi, (_, default_lbl) in enumerate(self._heads):
                per_sv = parts[hi]
                if per_sv is None or len(per_sv) != len(frames):
                    continue
                for i, sv_det in enumerate(per_sv):
                    merged[i].extend(
                        _sv_to_detections(
                            sv_det,
                            class_id_map=self._class_names,
                            default_label=default_lbl,
                            use_sv_class_names=use_sv_names,
                        )
                    )
            return merged
        for head, default_lbl in self._heads:
            per_sv = head.predict_batch(images_rgb)
            if len(per_sv) != len(frames):
                continue
            for i, sv_det in enumerate(per_sv):
                merged[i].extend(
                    _sv_to_detections(
                        sv_det,
                        class_id_map=self._class_names,
                        default_label=default_lbl,
                        use_sv_class_names=use_sv_names,
                    )
                )
        return merged
