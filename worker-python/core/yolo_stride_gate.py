"""YOLO-only **gate**: decide which frames run the full detector stack.

This module does **not** run YOLO itself. It answers ``should_run_yolo(frame_idx)`` so the
pipeline can skip expensive work on many frames.

**Why “gate”?**  
The gate is the policy layer that opens or closes how often we invoke YOLO (and thus
LP/OCR that depend on YOLO boxes). ``GATE_MODE=yolo`` (default in ``settings.py``) uses this
scheduler. ``GATE_MODE=off`` disables the stride gate for the **chunked** path (full YOLO
inside each time chunk).

**Coarse vs dense**

* **Coarse stride** (``coarse_stride`` / ``YOLO_COARSE_STRIDE``): when we are **not**
  inside a post-activity “dense” window, we only run YOLO on frames where
  ``frame_idx % coarse_stride == 0``. Example: stride 8 → run on frames 0, 8, 16, …
  Use roughly **5–10** when you want fewer idle checks.

* **Dense stride** (``dense_stride`` / ``YOLO_DENSE_STRIDE``): after YOLO sees a **person**
  or **vehicle** (caller reports ``has_activity``), we extend an **activity window** and
  run YOLO on frames where ``frame_idx % dense_stride == 0`` **while** ``frame_idx`` is
  inside that window. Example: stride **2** → every **other** frame (0, 2, 4, … relative
  to indices; combined with coarse, both can fire on the same frame — one YOLO call).

* **Dense window** (``dense_window_frames`` / derived from ``YOLO_DENSE_WINDOW_SEC`` × FPS):
  number of **frames** after a positive ``observe()`` during which the dense rule applies.
  ``YOLO_DENSE_WINDOW_SEC`` is wall-time in seconds of the video; multiply by FPS to get
  frames. New hits **extend** ``activity_until`` forward from the current frame index.

**Flow**  
Each frame: if ``should_run_yolo`` → run YOLO → ``observe(frame_idx, has_person_or_vehicle)``.
If we skip YOLO, do not call ``observe`` (no new evidence).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YoloStrideGateConfig:
    """Tuning bundle for :class:`YoloStrideGate` (usually built from ``settings.py`` / env)."""

    coarse_stride: int = 8
    """When idle: run YOLO once per this many frame indices (typical 5–10)."""

    dense_stride: int = 2
    """Inside the activity window: run YOLO every this many frames (2 = every other)."""

    dense_window_frames: int = 48
    """After ``observe(..., True)``, keep ``frame_idx <= activity_until`` for this many frames."""


class YoloStrideGate:
    """Coarse sampling by default; temporarily densify after person/vehicle hits."""

    def __init__(self, cfg: YoloStrideGateConfig) -> None:
        self._c = cfg
        self._activity_until = -1

    def should_run_yolo(self, frame_idx: int) -> bool:
        """True if this frame index should run YOLO (and then LP/OCR if the pipeline does so)."""
        coarse = (frame_idx % max(1, self._c.coarse_stride)) == 0
        in_window = frame_idx <= self._activity_until
        dense = in_window and ((frame_idx % max(1, self._c.dense_stride)) == 0)
        return coarse or dense

    def observe(self, frame_idx: int, has_activity: bool) -> None:
        """Call only on frames where YOLO actually ran; pass whether person/vehicle were found."""
        if has_activity:
            self._activity_until = frame_idx + max(1, self._c.dense_window_frames)
