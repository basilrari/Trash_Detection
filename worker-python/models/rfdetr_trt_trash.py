"""RF-DETR trash + cigarette heads via TensorRT ``.engine`` files (fixed NCHW batch).

Engines use static batch (e.g. 8) and input size (e.g. 672×672) baked in at export time.
Requires ``tensorrt`` and ``pycuda`` on the inference machine.

Set ``RF_DETR_TRT_TIMING=1`` to print per-chunk ``[TRT]`` timing (preprocess / forward / postprocess).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np
import supervision as sv
import torch

from core.types import Detection, FrameData
from models.base import TrashDetector
from models.trash_detector import _sv_to_detections

logger = logging.getLogger(__name__)


def _trt_timing_enabled() -> bool:
    v = os.environ.get("RF_DETR_TRT_TIMING", "").strip().lower()
    return v in ("1", "true", "yes", "on")


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
    box_offsets: np.ndarray,
    eng_w: int,
    eng_h: int,
    topk: int = 300,
) -> list[dict[str, np.ndarray]]:
    """CPU fallback post-process (same math as :func:`_post_process_gpu`).

    ``target_sizes`` is ``(B, 2)`` ``(img_h, img_w)`` per frame (original RGB shape).
    ``box_offsets`` is ``(B, 2)`` ``(ox, oy)`` such that engine-normalized xyxy maps to
    original pixels via ``x_orig = x_eng * eng_w + ox`` (and same for y with ``eng_h``).
    """
    with torch.inference_mode():
        bbox_t = torch.from_numpy(out_bbox)
        logits_t = torch.from_numpy(out_logits)
        sizes_t = torch.from_numpy(target_sizes.astype(np.float32))
        off_t = torch.from_numpy(np.ascontiguousarray(box_offsets.astype(np.float32)))
        prob = logits_t.sigmoid()
        flat = prob.view(prob.shape[0], -1)
        k = min(topk, flat.shape[1])
        topk_values, topk_indexes = torch.topk(flat, k, dim=1)
        topk_boxes = topk_indexes // logits_t.shape[2]
        labels = topk_indexes % logits_t.shape[2]
        boxes = _box_cxcywh_to_xyxy(bbox_t)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        _ = sizes_t  # kept for API symmetry / future use
        ox, oy = off_t.unbind(1)
        ew = float(eng_w)
        eh = float(eng_h)
        scale_fct = torch.tensor(
            [ew, eh, ew, eh], dtype=boxes.dtype, device=boxes.device
        ).view(1, 1, 4)
        off4 = torch.stack([ox, oy, ox, oy], dim=1).unsqueeze(1)
        boxes = boxes * scale_fct + off4
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


def _post_process_gpu(
    bbox_t: torch.Tensor,
    logits_t: torch.Tensor,
    sizes_t: torch.Tensor,
    offsets_t: torch.Tensor,
    eng_w: int,
    eng_h: int,
    topk: int = 300,
) -> list[dict[str, np.ndarray]]:
    """Top-k decode + box scaling on GPU; small per-frame arrays copied to CPU last."""
    with torch.inference_mode():
        if bbox_t.dtype != torch.float32:
            bbox_t = bbox_t.float()
        if logits_t.dtype != torch.float32:
            logits_t = logits_t.float()
        if sizes_t.dtype != torch.float32:
            sizes_t = sizes_t.float()
        if offsets_t.dtype != torch.float32:
            offsets_t = offsets_t.float()

        prob = logits_t.sigmoid()
        flat = prob.view(prob.shape[0], -1)
        k = min(topk, flat.shape[1])
        topk_values, topk_indexes = torch.topk(flat, k, dim=1)
        topk_boxes = topk_indexes // logits_t.shape[2]
        labels = topk_indexes % logits_t.shape[2]
        boxes = _box_cxcywh_to_xyxy(bbox_t)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        _ = sizes_t
        ox, oy = offsets_t.unbind(1)
        ew = float(eng_w)
        eh = float(eng_h)
        scale_fct = torch.tensor(
            [ew, eh, ew, eh], dtype=boxes.dtype, device=boxes.device
        ).view(1, 1, 4)
        off4 = torch.stack([ox, oy, ox, oy], dim=1).unsqueeze(1)
        boxes = boxes * scale_fct + off4

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
                "(e.g. export ONNX with the upstream ``rfdetr`` tooling, then ``trtexec``)."
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
        self._output_names: list[str] = []
        self.input_name: str | None = None
        self.batch = 0
        self.height = 0
        self.width = 0
        self._setup_bindings()

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)

        self._torch_stream: torch.cuda.Stream | None = None
        self._torch_out: dict[str, torch.Tensor] = {}
        # GPU post (D2D + torch on ExternalStream) is only safe on the thread that owns this
        # PyCUDA ``Stream`` (created here). ``ThreadPoolExecutor`` workers must use D2H + CPU post.
        self._gpu_post_ready = bool(torch.cuda.is_available())
        if self._gpu_post_ready:
            try:
                self._torch_stream = torch.cuda.ExternalStream(self.stream.handle)
            except Exception:
                logger.warning(
                    "Could not wrap PyCUDA stream for PyTorch; falling back to D2H + CPU post.",
                    exc_info=True,
                )
                self._torch_stream = None
                self._gpu_post_ready = False

        if self._gpu_post_ready and self._torch_stream is not None:
            dev = torch.device("cuda", torch.cuda.current_device())
            for name in self._output_names:
                buf = self.bindings[name]
                shape = buf["shape"]
                dt = buf["dtype"]
                if np.dtype(dt) == np.dtype(np.float16):
                    tdt = torch.float16
                else:
                    tdt = torch.float32
                self._torch_out[name] = torch.empty(shape, dtype=tdt, device=dev)

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
            self.bindings[name] = {
                "host": host,
                "device": device,
                "shape": shape,
                "dtype": dtype,
                "nbytes": int(host.nbytes),
            }
            self.context.set_tensor_address(name, int(device))
            if mode == trt.TensorIOMode.OUTPUT:
                self._output_names.append(name)
        if not self.input_name:
            raise RuntimeError("No TensorRT input tensor found.")

    @staticmethod
    def _orig_xy_offset_for_engine_tile(h: int, w: int, H: int, W: int) -> tuple[float, float]:
        """Original-image pixel ``(x, y)`` of engine input origin ``(0, 0)`` after center crop/pad.

        Maps engine-normalized coords to originals via ``x_orig = x_norm * W + ox`` (and ``y``).
        """
        if h >= H and w >= W:
            return ((w - W) / 2.0, (h - H) / 2.0)
        if h <= H and w <= W:
            return (-((W - w) / 2.0), -((H - h) / 2.0))
        if h >= H and w < W:
            return (-((W - w) / 2.0), (h - H) / 2.0)
        return ((w - W) / 2.0, -((H - h) / 2.0))

    def _rgb_to_engine_hw(self, rgb: np.ndarray) -> np.ndarray:
        """Fit ``rgb`` (H×W×3 RGB) to ``(self.height, self.width)`` by **center crop** and/or
        **zero pad** only — no ``cv2.resize`` / interpolation (different from typical training
        letterbox, but matches “no resizing” request).
        """
        H, W = self.height, self.width
        h, w = rgb.shape[:2]
        out = np.zeros((H, W, 3), dtype=rgb.dtype)
        if h >= H and w >= W:
            y0 = (h - H) // 2
            x0 = (w - W) // 2
            return rgb[y0 : y0 + H, x0 : x0 + W].copy()
        if h <= H and w <= W:
            y0 = (H - h) // 2
            x0 = (W - w) // 2
            out[y0 : y0 + h, x0 : x0 + w] = rgb
            return out
        if h >= H and w < W:
            y0 = (h - H) // 2
            slab = rgb[y0 : y0 + H, :]  # (H, w, 3)
            x0 = (W - w) // 2
            out[:, x0 : x0 + w] = slab
            return out
        # h < H and w >= W
        x0 = (w - W) // 2
        slab = rgb[:, x0 : x0 + W]  # (h, W, 3)
        y0 = (H - h) // 2
        out[y0 : y0 + h, :] = slab
        return out

    def _preprocess(self, frames_bgr: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        CPU preprocessing without ``cv2.resize``: center **crop** and/or **pad** to engine
        ``self.height`` × ``self.width``, then batched normalize to NCHW float32.
        ``sizes`` is ``(B, 2)`` ``(img_h, img_w)``; ``box_offsets`` is ``(B, 2)`` ``(ox, oy)``
        for :func:`_post_process` / :func:`_post_process_gpu`.
        """
        if not frames_bgr:
            return (
                np.empty((0, 3, self.height, self.width), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
            )

        tiles: list[np.ndarray] = []
        sizes_list: list[tuple[int, int]] = []
        off_list: list[tuple[float, float]] = []
        H, W = self.height, self.width

        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h0, w0 = rgb.shape[:2]
            sizes_list.append((h0, w0))
            off_list.append(self._orig_xy_offset_for_engine_tile(h0, w0, H, W))
            tiles.append(self._rgb_to_engine_hw(rgb))

        batch_rgb = np.stack(tiles, axis=0)  # (B, H, W, 3)
        batch = batch_rgb.astype(np.float32) / 255.0
        batch = (batch - self.mean) / self.std
        batch = np.transpose(batch, (0, 3, 1, 2))  # (B, 3, H, W)

        sizes = np.asarray(sizes_list, dtype=np.float32)
        offsets = np.asarray(off_list, dtype=np.float32)

        return batch.astype(np.float32), sizes, offsets

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

    def _pick_outputs_torch(self) -> tuple[torch.Tensor, torch.Tensor]:
        outs = [self._torch_out[n] for n in self._output_names]
        if len(outs) < 2:
            raise RuntimeError(f"Expected 2 output tensors, got {len(outs)}")
        a, b = outs[0], outs[1]
        if a.shape[-1] == 4 and b.shape[-1] != 4:
            return a, b
        if b.shape[-1] == 4 and a.shape[-1] != 4:
            return b, a
        return outs[0], outs[1]

    def _use_gpu_torch_decode_this_thread(self) -> bool:
        """GPU decode is only valid on the thread that created the PyCUDA ``Stream``."""
        return (
            self._gpu_post_ready
            and self._torch_stream is not None
            and threading.current_thread() is threading.main_thread()
        )

    def _infer_raw_async(self, batch_nchw: np.ndarray) -> None:
        """Queue H2D + execute + D2D into torch GPU buffers on ``self.stream`` (no synchronize)."""
        inp = self.bindings[self.input_name]
        if tuple(batch_nchw.shape) != tuple(inp["shape"]):
            raise ValueError(f"Expected input shape {inp['shape']}, got {tuple(batch_nchw.shape)}")
        np.copyto(inp["host"], batch_nchw.ravel().astype(inp["dtype"], copy=False))
        self.cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)
        if not self.context.execute_async_v3(self.stream.handle):
            raise RuntimeError("TensorRT execute_async_v3 failed.")
        if self._use_gpu_torch_decode_this_thread():
            for name, buf in self.bindings.items():
                if name == self.input_name:
                    continue
                dst = self._torch_out[name]
                if dst.numel() * dst.element_size() != buf["nbytes"]:
                    raise RuntimeError(f"Torch buffer size mismatch for {name}")
                self.cuda.memcpy_dtod_async(
                    int(dst.data_ptr()),
                    int(buf["device"]),
                    buf["nbytes"],
                    self.stream,
                )
        else:
            for name, buf in self.bindings.items():
                if name == self.input_name:
                    continue
                self.cuda.memcpy_dtoh_async(buf["host"], buf["device"], self.stream)

    def _infer_raw_numpy_outputs(self, batch_nchw: np.ndarray) -> list[np.ndarray]:
        """D2H to pagelocked host + one stream sync (fallback path)."""
        self._infer_raw_async(batch_nchw)
        self.stream.synchronize()
        outputs: list[np.ndarray] = []
        for name, buf in self.bindings.items():
            if name == self.input_name:
                continue
            outputs.append(buf["host"].reshape(buf["shape"]).copy())
        return outputs

    def decode_batch_nchw(
        self,
        batch_nchw: np.ndarray,
        sizes: np.ndarray,
        box_offsets: np.ndarray,
        valid: int,
        *,
        top_n: int = 50,
        preprocess_ms: float | None = None,
        timing_label: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run TRT + postprocess on a pre-built NCHW batch (``valid`` first rows are real frames).

        ``preprocess_ms`` is optional (for timing logs when the caller timed preprocess separately).
        ``timing_label`` prefixes ``[TRT]`` logs (e.g. ``trash`` / ``cigarette``) when two heads run.
        """
        t_fwd0 = time.perf_counter()
        eng_w, eng_h = int(self.width), int(self.height)
        if self._use_gpu_torch_decode_this_thread():
            self._infer_raw_async(batch_nchw)
            # Same CUDA stream as PyCUDA: run GPU post without an extra pycuda stream.synchronize().
            with torch.cuda.stream(self._torch_stream):
                dets_t, logits_t = self._pick_outputs_torch()
                sizes_t = torch.as_tensor(
                    np.ascontiguousarray(sizes.astype(np.float32)),
                    device=dets_t.device,
                    dtype=torch.float32,
                )
                off_t = torch.as_tensor(
                    np.ascontiguousarray(box_offsets.astype(np.float32)),
                    device=dets_t.device,
                    dtype=torch.float32,
                )
                rows = _post_process_gpu(dets_t, logits_t, sizes_t, off_t, eng_w, eng_h)
            forward_ms = (time.perf_counter() - t_fwd0) * 1000.0
            t_post0 = time.perf_counter()
            # Threshold / pack (CPU, small arrays)
            out = self._pack_detections(rows, valid, top_n)
            post_ms = (time.perf_counter() - t_post0) * 1000.0
        else:
            raw_outs = self._infer_raw_numpy_outputs(batch_nchw)
            forward_ms = (time.perf_counter() - t_fwd0) * 1000.0
            t_post0 = time.perf_counter()
            dets, logits = self._pick_outputs(raw_outs)
            rows = _post_process(dets, logits, sizes, box_offsets, eng_w, eng_h)
            out = self._pack_detections(rows, valid, top_n)
            post_ms = (time.perf_counter() - t_post0) * 1000.0

        if _trt_timing_enabled():
            bs = int(valid)
            tag = f"{timing_label} | " if timing_label else ""
            if preprocess_ms is None:
                print(
                    f"[TRT] {tag}batch_size={bs} | forward={forward_ms:.1f}ms | postprocess={post_ms:.1f}ms"
                )
            else:
                print(
                    f"[TRT] {tag}batch_size={bs} | preprocess={float(preprocess_ms):.1f}ms | "
                    f"forward={forward_ms:.1f}ms | postprocess={post_ms:.1f}ms"
                )
        return out

    def _pack_detections(
        self,
        rows: list[dict[str, np.ndarray]],
        valid: int,
        top_n: int,
    ) -> list[dict[str, Any]]:
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
        t0 = time.perf_counter()
        out: list[dict[str, Any]] = []
        for start in range(0, len(frames_bgr), self.batch):
            chunk = frames_bgr[start : start + self.batch]
            valid = len(chunk)
            if valid < self.batch:
                chunk = chunk + [np.zeros_like(chunk[0]) for _ in range(self.batch - valid)]
            t_pre0 = time.perf_counter()
            batch_nchw, sizes, box_offsets = self._preprocess(chunk)
            preprocess_ms = (time.perf_counter() - t_pre0) * 1000.0
            out.extend(
                self.decode_batch_nchw(
                    batch_nchw,
                    sizes,
                    box_offsets,
                    valid,
                    top_n=top_n,
                    preprocess_ms=preprocess_ms,
                )
            )
        n = len(frames_bgr)
        total_ms = (time.perf_counter() - t0) * 1000.0
        fps = (n * 1000.0 / total_ms) if total_ms > 0 else 0.0
        print(
            f"[RF-DETR] engine | batch_size={n} | total_ms={total_ms:.1f} | fps={fps:.1f}"
        )
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

    Each batch is **preprocessed once** (BGR→tile→NCHW on the trash head's wrapper); both heads
    then run ``decode_batch_nchw`` on that shared tensor **in parallel** (two threads; one TRT
    forward + post per head).
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
        """Fixed engine input ``(H, W)`` (e.g. 672, 672); frames are center-cropped/padded to this."""
        h0 = self._heads[0][0].height
        w0 = self._heads[0][0].width
        return (h0, w0)

    def detect_trash(self, frames: List[FrameData]) -> List[List[Detection]]:
        if not frames:
            return []
        t0 = time.perf_counter()
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
            t_pre0 = time.perf_counter()
            batch_nchw, sizes, box_offsets = primary._preprocess(padded)
            preprocess_ms = (time.perf_counter() - t_pre0) * 1000.0
            if len(self._heads) > 1:
                parts: list[list[dict[str, Any]] | None] = [None] * len(self._heads)
                with ThreadPoolExecutor(max_workers=len(self._heads)) as ex:
                    futures: dict[Future, int] = {}
                    for hi in range(len(self._heads)):
                        label = self._heads[hi][1]
                        fut = ex.submit(
                            self._heads[hi][0].decode_batch_nchw,
                            batch_nchw,
                            sizes,
                            box_offsets,
                            valid,
                            top_n=50,
                            preprocess_ms=preprocess_ms if hi == 0 else None,
                            timing_label=label,
                        )
                        futures[fut] = hi
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
                for hi, (head, default_lbl) in enumerate(self._heads):
                    per = head.decode_batch_nchw(
                        batch_nchw,
                        sizes,
                        box_offsets,
                        valid,
                        top_n=50,
                        preprocess_ms=preprocess_ms if hi == 0 else None,
                        timing_label=default_lbl,
                    )
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
        n = len(frames)
        total_ms = (time.perf_counter() - t0) * 1000.0
        fps = (n * 1000.0 / total_ms) if total_ms > 0 else 0.0
        print(
            f"[RF-DETR] trash+cigarette | batch_size={n} | total_ms={total_ms:.1f} | fps={fps:.1f}"
        )
        return merged


# Timing: export RF_DETR_TRT_TIMING=1 to print ``[TRT]`` lines from ``decode_batch_nchw`` /
# ``predict_batch``. With two heads, expect two ``[TRT] <head> | ...`` lines per batch (shared
# preprocess is printed only on the trash line).
