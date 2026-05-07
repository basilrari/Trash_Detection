"""RF-DETR litter / cigarette heads (``rfdetr`` + local ``.pth`` checkpoints)."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping

import cv2
import numpy as np
import torch

from core.types import Detection, FrameData
from models.base import TrashDetector

logger = logging.getLogger(__name__)


def _ckpt_get(args_obj: Any, key: str) -> Any:
    if args_obj is None:
        return None
    if isinstance(args_obj, Mapping):
        return args_obj.get(key)
    return getattr(args_obj, key, None)


def _load_checkpoint_dict(weights_path: str) -> Dict[str, Any] | None:
    p = Path(weights_path)
    if not p.is_file():
        return None
    try:
        try:
            return torch.load(str(p), map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(str(p), map_location="cpu")
    except Exception:
        return None


def _rfdetr_kwargs_from_ckpt_data(ckpt: Dict[str, Any], cfg_class: type[Any]) -> Dict[str, Any]:
    """Pull constructor kwargs from ``checkpoint['args']`` that exist on the model config class."""
    allowed = frozenset(cfg_class.model_fields.keys()) - {"pretrain_weights"}
    raw_args = ckpt.get("args")
    out: Dict[str, Any] = {}
    for key in allowed:
        val = _ckpt_get(raw_args, key)
        if val is None:
            continue
        out[key] = val
    return out


def _infer_pe_side_from_position_embeddings(ckpt: Dict[str, Any]) -> int | None:
    """Infer ``positional_encoding_size`` (square grid side) from backbone ``position_embeddings`` shape.

    RF-DETR windowed ViT uses ``num_patches = side * side`` patch tokens plus one class token.
    ``positional_encoding_size`` in config satisfies ``implied_resolution = PE * patch_size`` and
    ``num_patches = PE ** 2`` when the grid is square — see ``rfdetr`` ``dinov2.py`` (``implied_resolution``).
    """
    model = ckpt.get("model")
    if not isinstance(model, dict):
        return None
    for _key, tensor in model.items():
        if "position_embeddings" not in _key or not isinstance(tensor, torch.Tensor):
            continue
        if tensor.dim() != 3:
            continue
        seq_len = int(tensor.shape[1])
        n_patch = seq_len - 1
        if n_patch < 1:
            continue
        side = int(round(math.sqrt(n_patch)))
        if side * side == n_patch:
            return side
    return None


def _default_patch_size(cfg_class: type[Any]) -> int:
    field = cfg_class.model_fields.get("patch_size")
    d = getattr(field, "default", None) if field is not None else None
    return int(d) if isinstance(d, int) else 16


def _snap_side_to_block_grid(resolution: int, patch_size: int, num_windows: int) -> int:
    """``rfdetr`` backbone requires H,W divisible by ``patch_size * num_windows`` (see ``predict()``)."""
    block = int(patch_size) * int(num_windows)
    if block <= 0:
        return int(resolution)
    r = int(resolution)
    if r % block == 0:
        return r
    low = (r // block) * block
    high = low + block
    # Prefer rounding up on ties so we do not shrink below the trained default more than necessary.
    return high if high - r <= r - low else max(low, block)


def _predict_kwargs_if_needed(rfdetr_model: Any) -> Dict[str, Any]:
    """Pass ``shape=`` to ``predict`` when ``model.resolution`` is not a multiple of ``patch_size * num_windows``."""
    cfg = rfdetr_model.model_config
    res = int(rfdetr_model.model.resolution)
    patch = int(cfg.patch_size)
    nw = int(getattr(cfg, "num_windows", 2))
    snapped = _snap_side_to_block_grid(res, patch, nw)
    if snapped == res:
        return {}
    return {"shape": (snapped, snapped)}


def _manual_rfdetr_overrides() -> tuple[Dict[str, Any], frozenset[str]]:
    from settings import (
        RF_DETR_NUM_CLASSES,
        RF_DETR_PATCH_SIZE,
        RF_DETR_POSITIONAL_ENCODING_SIZE,
        RF_DETR_RESOLUTION,
    )

    out: Dict[str, Any] = {}
    if RF_DETR_PATCH_SIZE is not None:
        out["patch_size"] = RF_DETR_PATCH_SIZE
    if RF_DETR_NUM_CLASSES is not None:
        out["num_classes"] = RF_DETR_NUM_CLASSES
    if RF_DETR_RESOLUTION is not None:
        out["resolution"] = RF_DETR_RESOLUTION
    if RF_DETR_POSITIONAL_ENCODING_SIZE is not None:
        out["positional_encoding_size"] = RF_DETR_POSITIONAL_ENCODING_SIZE
    return out, frozenset(out.keys())


def _sv_to_detections(
    sv_det: Any,
    *,
    class_id_map: Dict[int, str] | None,
    default_label: str,
) -> List[Detection]:
    """Convert ``supervision.Detections`` to our :class:`Detection` list."""
    if sv_det is None or len(sv_det) == 0:
        return []

    xyxy = sv_det.xyxy
    confs = sv_det.confidence
    cls_ids = sv_det.class_id
    names = None
    try:
        names = sv_det.data.get("class_name") if hasattr(sv_det, "data") else None
    except Exception:
        names = None

    out: List[Detection] = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = map(float, xyxy[i])
        cf = float(confs[i]) if confs is not None else 0.0
        cid = int(cls_ids[i]) if cls_ids is not None else 0
        if class_id_map and cid in class_id_map:
            label = class_id_map[cid]
        elif names is not None and i < len(names):
            label = str(names[i]).strip() or default_label
        else:
            label = default_label
        out.append(Detection(bbox=(x1, y1, x2, y2), label=label, confidence=cf))
    return out


def _ensure_positional_encoding_size(merged: Dict[str, Any]) -> None:
    """If ``resolution`` and ``patch_size`` are set but PE is not, set ``PE = resolution // patch_size``."""
    if merged.get("positional_encoding_size") is not None:
        return
    res = merged.get("resolution")
    ps = merged.get("patch_size")
    if res is None or ps is None:
        return
    merged["positional_encoding_size"] = int(res) // int(ps)


def _rfdetr_size_hint_from_checkpoint(ckpt: Dict[str, Any]) -> str | None:
    """Return ``nano`` \| ``small`` \| ``medium`` \| ``large`` if the checkpoint records it."""
    raw = ckpt.get("args")
    if raw is None:
        return None
    for key in ("size", "model_size", "rfdetr_size", "rf_detr_size"):
        v = _ckpt_get(raw, key)
        if v is None:
            continue
        if isinstance(v, str):
            s = v.strip().lower()
            for prefix in ("rfdetr-", "rf-detr-", "rfdetr_", "rf_detr_"):
                if s.startswith(prefix):
                    s = s[len(prefix) :]
                    break
            s = s.split(".")[0].split("/")[0].strip("-_")
            if s in ("nano", "small", "medium", "large"):
                return s
    return None


def _rfdetr_ctor_candidates(hint: str | None) -> List[Any]:
    """Prefer checkpoint hint; otherwise try common families until one loads."""
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    table = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }
    if hint and hint in table:
        return [table[hint]]
    return [RFDETRMedium, RFDETRSmall, RFDETRLarge, RFDETRNano]


def _merge_init_kwargs_for_rfdetr(ctor: type[Any], weights_path: str, ckpt: Dict[str, Any]) -> Dict[str, Any]:
    cfg_class = ctor._model_config_class
    merged: Dict[str, Any] = dict(_rfdetr_kwargs_from_ckpt_data(ckpt, cfg_class))
    manual, manual_keys = _manual_rfdetr_overrides()
    merged.update(manual)

    inferred_side = _infer_pe_side_from_position_embeddings(ckpt) if ckpt else None
    if inferred_side is not None and "positional_encoding_size" not in manual_keys:
        ps = int(merged["patch_size"]) if merged.get("patch_size") is not None else _default_patch_size(cfg_class)
        merged["positional_encoding_size"] = inferred_side
        if "resolution" not in manual_keys:
            merged["resolution"] = inferred_side * ps
    else:
        _ensure_positional_encoding_size(merged)

    merged["pretrain_weights"] = str(weights_path)
    return merged


def _optimize_rfdetr_for_inference(model: Any) -> None:
    """Use rfdetr's export/inference path so ``predict`` skips the unoptimized warning.

    ``compile=False`` keeps variable batch sizes valid (we call ``predict`` with
    ``len(frames)`` images). If we snap ``predict(..., shape=...)``, align
    ``model.model.resolution`` first so the optimized tensor size matches.
    """
    optimize = getattr(model, "optimize_for_inference", None)
    if not callable(optimize):
        return
    ctx = model.model
    orig_res = int(ctx.resolution)
    pkw = _predict_kwargs_if_needed(model)
    shape = pkw.get("shape")
    if shape is not None:
        h, w = int(shape[0]), int(shape[1])
        if h != w:
            return
        ctx.resolution = h
    try:
        optimize(compile=False, batch_size=1, dtype=torch.float32)
    except Exception:
        ctx.resolution = orig_res
        logger.warning("RF-DETR optimize_for_inference failed; using unoptimized path.", exc_info=True)


def _build_rfdetr(weights_path: str) -> Any:
    """Load **your** ``.pth`` only — pick ``RFDETR*`` class from checkpoint metadata or by trial."""
    ckpt = _load_checkpoint_dict(weights_path) or {}
    hint = _rfdetr_size_hint_from_checkpoint(ckpt)
    errors: list[str] = []
    for ctor in _rfdetr_ctor_candidates(hint):
        try:
            merged = _merge_init_kwargs_for_rfdetr(ctor, weights_path, ckpt)
            model = ctor(**merged)
            _optimize_rfdetr_for_inference(model)
            return model
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{ctor.__name__}: {exc}")
            continue
    raise RuntimeError(
        f"Could not load RF-DETR checkpoint {weights_path!r} (hint={hint!r}). "
        "Attempts:\n  " + "\n  ".join(errors)
    )


class RfDetrTrashDetector(TrashDetector):
    """Runs **your** two RF-DETR checkpoints: ``trash.pth`` and ``cigarette.pth`` (both required).

    The ``rfdetr`` size class (Nano/Small/Medium/Large) is inferred per file from checkpoint
    ``args`` when possible, otherwise by trying families until weights load.

    Expects RGB inference internally; ``detect_trash`` converts BGR ``FrameData`` images.
    """

    def __init__(
        self,
        trash_weights: str | Path,
        cigarette_weights: str | Path,
        *,
        class_names: Dict[int, str] | None = None,
        conf_threshold: float = 0.4,
    ) -> None:
        self._conf = float(conf_threshold)
        self._class_names = dict(class_names) if class_names else None
        tw = Path(trash_weights)
        cw = Path(cigarette_weights)
        if not tw.is_file():
            raise FileNotFoundError(f"RF-DETR trash weights not found: {tw}")
        if not cw.is_file():
            raise FileNotFoundError(f"RF-DETR cigarette weights not found: {cw}")
        if tw.resolve() == cw.resolve():
            raise ValueError("trash.pth and cigarette.pth must be two different checkpoint files")

        self._models: List[tuple[Any, str, str]] = [
            (_build_rfdetr(str(tw)), "trash", "trash"),
            (_build_rfdetr(str(cw)), "cigarette", "cigarette"),
        ]

    def detect_trash(self, frames: List[FrameData]) -> List[List[Detection]]:
        if not frames:
            return []
        images_rgb = [cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB) for f in frames]
        merged: List[List[Detection]] = [[] for _ in frames]

        for model, default_lbl, _tag in self._models:
            pkw = _predict_kwargs_if_needed(model)
            raw = model.predict(images_rgb, threshold=self._conf, **pkw)
            per_frame = raw if isinstance(raw, list) else [raw]
            if len(per_frame) != len(frames):
                continue
            for i, sv_det in enumerate(per_frame):
                merged[i].extend(
                    _sv_to_detections(
                        sv_det,
                        class_id_map=self._class_names,
                        default_label=default_lbl,
                    )
                )
        return merged
