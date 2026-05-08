"""RF-DETR trash + cigarette heads via TensorRT ``.engine`` files (fixed NCHW batch).

Engines are built with static batch (e.g. 8) and input size (e.g. 672×672); see export notes.
Requires ``tensorrt`` and ``pycuda`` on the inference machine.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np
import supervision as sv
import torch

from core.types import Detection, FrameData
from models.base import TrashDetector
from models.trash_detector import _parallel_rfdetr_heads_enabled, _sv_to_detections


def _box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack(
        [
            x_c - 0.5 * w.clamp(min=0.0),
            y_c - 0.5 * h.clamp(min=0.0),
            x_c + 0.5 * w.clamp(min=0.0),
            y_c + 0.5 * h.clamp(min=0.0),
        ],
        dim=-1,
    )


def _post_process(
    out_bbox: np.ndarray,
    out_logits: np.ndarray,
    target_sizes: np.ndarray,
    topk: int = 300,
) -> list[dict[str, np.ndarray]]:
    # inference_mode avoids autograd bookkeeping on the hot path.
    with torch.inference_mode():
        bbox_t = torch.from_numpy(out_bbox)
        logits_t = torch.from_numpy(out_logits)
        sizes_t = torch.from_numpy(target_sizes.astype(np.float32))
        prob = logits_t.sigmoid()
        flat = prob.view(prob.shape[0], -1)
        k = min(topk, flat.shape[1])
        topk_values, topk_indexes = torch.topk(flat, k, dim=1)
        topk_boxes = topk_indexes // logits_t.shape[2]
        labels = topk_indexes % logits_t.shape[2]
        boxes = _box_cxcywh_to_xyxy(bbox_t)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        img_h, img_w = sizes_t.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        out: list[dict[str, np.ndarray]] = []
        for score, label, box in zip(topk_values, labels, boxes):
            out.append(
                {
                    "scores": score.detach().cpu().numpy(),
                    "labels": label.detach().cpu().numpy().astype(np.int64),
                    "boxes": box.detach().cpu().numpy(),
                }
            )
        return out


def _tensorrt_deserialize_hint(engine_path: str | Path) -> str:
    try:
        import torch

        if torch.cuda.is_available():
            c = torch.cuda.get_device_capability(0)
            n = torch.cuda.get_device_name(0)
            vis = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
            return (
                "\nTensorRT engines are tied to the GPU **architecture** they were built on "
                "(e.g. sm_86 vs sm_120). This plan does not match the GPU visible to this process.\n"
                f"  torch.cuda:0 = {n!r}  sm_{c[0]}{c[1]}  (CUDA_VISIBLE_DEVICES={vis})\n"
                "Fix: set CUDA_VISIBLE_DEVICES to the same physical GPU used when building the .engine "
                "files, or rebuild trash.engine and cigarette.engine on this machine/GPU "
                "(see scripts/export_rfdetr_heads.py + trtexec)."
            )
    except Exception:
        pass
    return (
        "\nTensorRT engine load failed. Re-export or rebuild the .engine for the GPU "
        "you use at inference time."
    )


class TensorRTEngineWrapper:
    """Loads a fixed-shape RF-DETR TensorRT engine and runs batched inference."""

    def __init__(self, engine_path: str | Path, conf_threshold: float = 0.4) -> None:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        import tensorrt as trt

        self.cuda = cuda
        self.trt = trt
        self.conf = float(conf_threshold)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        blob = Path(engine_path).read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine: {engine_path}\n{_tensorrt_deserialize_hint(engine_path)}"
            )
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings: dict[str, dict[str, Any]] = {}
        self.input_name: str | None = None
        self.batch = 0
        self.height = 0
        self.width = 0
        self._setup_bindings()

    def _setup_bindings(self) -> None:
        trt = self.trt
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            if mode == trt.TensorIOMode.INPUT and -1 in shape:
                raise ValueError("Dynamic input engine is not supported by this wrapper.")
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.batch = int(shape[0])
                self.height = int(shape[2])
                self.width = int(shape[3])
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape))
            host = self.cuda.pagelocked_empty(size, dtype)
            device = self.cuda.mem_alloc(host.nbytes)
            self.bindings[name] = {"host": host, "device": device, "shape": shape, "dtype": dtype}
            self.context.set_tensor_address(name, int(device))
        if not self.input_name:
            raise RuntimeError("No TensorRT input tensor found.")

    def _preprocess(self, frames_bgr: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        rows = []
        sizes = []
        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h0, w0 = rgb.shape[:2]
            sizes.append((h0, w0))
            resized = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            x = resized.astype(np.float32) / 255.0
            x = (x - mean) / std
            x = np.transpose(x, (2, 0, 1))
            rows.append(x)
        return np.stack(rows, axis=0).astype(np.float32), np.asarray(sizes, dtype=np.float32)

    @staticmethod
    def _pick_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if len(outputs) < 2:
            raise RuntimeError(f"Expected 2 outputs, got {len(outputs)}")
        a, b = outputs[0], outputs[1]
        if a.shape[-1] == 4 and b.shape[-1] != 4:
            return a, b
        if b.shape[-1] == 4 and a.shape[-1] != 4:
            return b, a
        return outputs[0], outputs[1]

    def _infer_raw(self, batch_nchw: np.ndarray) -> list[np.ndarray]:
        inp = self.bindings[self.input_name]
        if tuple(batch_nchw.shape) != tuple(inp["shape"]):
            raise ValueError(f"Expected input shape {inp['shape']}, got {tuple(batch_nchw.shape)}")
        np.copyto(inp["host"], batch_nchw.ravel().astype(inp["dtype"], copy=False))
        self.cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)
        if not self.context.execute_async_v3(self.stream.handle):
            raise RuntimeError("TensorRT execute_async_v3 failed.")
        for name, buf in self.bindings.items():
            if name == self.input_name:
                continue
            self.cuda.memcpy_dtoh_async(buf["host"], buf["device"], self.stream)
        self.stream.synchronize()
        outputs: list[np.ndarray] = []
        for name, buf in self.bindings.items():
            if name == self.input_name:
                continue
            # One contiguous copy off the pagelocked D2H staging buffer before the next enqueue.
            outputs.append(buf["host"].reshape(buf["shape"]).copy())
        return outputs

    def decode_batch_nchw(
        self,
        batch_nchw: np.ndarray,
        sizes: np.ndarray,
        valid: int,
        *,
        top_n: int = 50,
    ) -> list[dict[str, Any]]:
        """Run TRT + postprocess on a pre-built NCHW batch (``valid`` first rows are real frames)."""
        raw_outs = self._infer_raw(batch_nchw)
        dets, logits = self._pick_outputs(raw_outs)
        rows = _post_process(dets, logits, sizes)
        out: list[dict[str, Any]] = []
        for i in range(valid):
            scores = rows[i]["scores"]
            boxes = rows[i]["boxes"]
            labels = rows[i]["labels"]
            mask = scores >= self.conf
            scores = scores[mask]
            boxes = boxes[mask]
            labels = labels[mask]
            if scores.size == 0:
                out.append({"boxes": [], "scores": [], "labels": []})
                continue
            order = np.argsort(-scores)[:top_n]
            out.append(
                {
                    "boxes": boxes[order].astype(float).tolist(),
                    "scores": scores[order].astype(float).tolist(),
                    "labels": labels[order].astype(int).tolist(),
                }
            )
        return out

    def predict_batch(self, frames_bgr: list[np.ndarray], top_n: int = 50) -> list[dict[str, Any]]:
        if not frames_bgr:
            return []
        out: list[dict[str, Any]] = []
        for start in range(0, len(frames_bgr), self.batch):
            chunk = frames_bgr[start : start + self.batch]
            valid = len(chunk)
            if valid < self.batch:
                chunk = chunk + [np.zeros_like(chunk[0]) for _ in range(self.batch - valid)]
            batch_nchw, sizes = self._preprocess(chunk)
            out.extend(self.decode_batch_nchw(batch_nchw, sizes, valid, top_n=top_n))
        return out


def _dict_to_sv(d: dict[str, Any]) -> sv.Detections:
    boxes = d.get("boxes") or []
    scores = d.get("scores") or []
    labels = d.get("labels") or []
    if not boxes:
        return sv.Detections(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            confidence=np.zeros((0,), dtype=np.float32),
            class_id=np.zeros((0,), dtype=np.int64),
        )
    xyxy = np.asarray(boxes, dtype=np.float32)
    conf = np.asarray(scores, dtype=np.float32)
    cid = np.asarray(labels, dtype=np.int64)
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cid)


class RfDetrTrtTrashDetector(TrashDetector):
    """Two TensorRT RF-DETR engines (trash + cigarette); same ``detect_trash`` contract as PyTorch.

    When ``_parallel_rfdetr_heads_enabled()`` is true (default), both heads decode the same
    NCHW batch on worker threads. Set ``RF_DETR_PARALLEL_HEADS=0`` to run heads strictly
    one after the other.
    """

    def __init__(
        self,
        trash_engine: str | Path,
        cigarette_engine: str | Path,
        *,
        class_names: dict[int, str] | None = None,
        conf_threshold: float = 0.4,
    ) -> None:
        self._class_names = dict(class_names) if class_names else None
        use_sv = class_names is not None
        self._heads: List[Tuple[TensorRTEngineWrapper, str]] = [
            (
                TensorRTEngineWrapper(Path(trash_engine), conf_threshold=conf_threshold),
                "trash",
            ),
            (
                TensorRTEngineWrapper(Path(cigarette_engine), conf_threshold=conf_threshold),
                "cigarette",
            ),
        ]
        self._use_sv_names = use_sv

    @property
    def engine_batch_size(self) -> int:
        """Static batch baked into the engines (both heads must match)."""
        return int(self._heads[0][0].batch)

    @property
    def engine_input_hw(self) -> tuple[int, int]:
        """Model input (H, W) after internal resize (e.g. 672, 672)."""
        h0 = self._heads[0][0].height
        w0 = self._heads[0][0].width
        return (h0, w0)

    def detect_trash(self, frames: List[FrameData]) -> List[List[Detection]]:
        if not frames:
            return []
        images_bgr = [f.image for f in frames]
        merged: List[List[Detection]] = [[] for _ in frames]
        # Preprocess each engine batch once: both heads share the same NCHW input (same export H×W×B).
        primary = self._heads[0][0]
        B = int(primary.batch)
        for start in range(0, len(images_bgr), B):
            chunk_valid = images_bgr[start : start + B]
            valid = len(chunk_valid)
            if valid < B:
                padded = chunk_valid + [np.zeros_like(chunk_valid[0]) for _ in range(B - valid)]
            else:
                padded = chunk_valid
            batch_nchw, sizes = primary._preprocess(padded)
            if _parallel_rfdetr_heads_enabled() and len(self._heads) > 1:
                parts: list[list[dict[str, Any]] | None] = [None] * len(self._heads)
                with ThreadPoolExecutor(max_workers=len(self._heads)) as ex:
                    futures = {
                        ex.submit(
                            self._heads[hi][0].decode_batch_nchw,
                            batch_nchw,
                            sizes,
                            valid,
                            top_n=50,
                        ): hi
                        for hi in range(len(self._heads))
                    }
                    for fut in as_completed(futures):
                        hi = futures[fut]
                        parts[hi] = fut.result()
                for hi, (_, default_lbl) in enumerate(self._heads):
                    per = parts[hi]
                    if per is None or len(per) != valid:
                        continue
                    for j, d in enumerate(per):
                        i = start + j
                        sv_det = _dict_to_sv(d)
                        merged[i].extend(
                            _sv_to_detections(
                                sv_det,
                                class_id_map=self._class_names,
                                default_label=default_lbl,
                                use_sv_class_names=self._use_sv_names,
                            )
                        )
            else:
                for head, default_lbl in self._heads:
                    per = head.decode_batch_nchw(batch_nchw, sizes, valid, top_n=50)
                    if len(per) != valid:
                        continue
                    for j, d in enumerate(per):
                        i = start + j
                        sv_det = _dict_to_sv(d)
                        merged[i].extend(
                            _sv_to_detections(
                                sv_det,
                                class_id_map=self._class_names,
                                default_label=default_lbl,
                                use_sv_class_names=self._use_sv_names,
                            )
                        )
        return merged
