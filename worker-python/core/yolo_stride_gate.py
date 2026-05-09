"""YOLO-only **gate**: decide which frames run the full detector stack.

This module does **not** run YOLO itself. It answers ``should_run_yolo(frame_idx)`` so the
pipeline can skip expensive work on many frames — used when ``settings.GATE_MODE == "yolo"``.

**Numeric ``GATE_MODE`` (``"1"``, ``"2"``, …)**  
Uniform stride + micro-batching is implemented in ``pipelines.test_pipeline`` (not here):
scene YOLO on ``frame_idx % N == 0`` only, with ``YOLO_MICRO_BATCH_SIZE`` batched ``detect()`` calls.

**Why “gate”?**  
``GATE_MODE=yolo`` (default in older configs) uses this scheduler. ``GATE_MODE=off`` disables the
stride gate for the **chunked** path (full YOLO inside each time chunk).

**Coarse vs dense** (``YoloStrideGate`` only)

* **Coarse stride** (``coarse_stride`` / ``YOLO_COARSE_STRIDE``): when we are **idle**
  (not in dense mode), we only run YOLO on frames where ``frame_idx % coarse_stride == 0``.
  Example: stride 8 → frames 0, 8, 16, …

* **Dense stride** (``dense_stride`` / ``YOLO_DENSE_STRIDE``): while **dense mode** is on
  (after a person/vehicle was seen), run YOLO on frames where ``frame_idx % dense_stride == 0``
  in addition to coarse hits (either branch can trigger a run on the same frame → one YOLO).

* **Leaving dense mode** — **not** a fixed wall-clock window. After YOLO sees a person or
  vehicle, we turn dense mode **on**. Each time YOLO runs and sees **no** person/vehicle, we
  increment a counter. When that counter reaches ``dense_idle_miss_streak`` (default **10**,
  ``YOLO_DENSE_IDLE_MISS_STREAK``), we turn dense mode **off** and go back to coarse-only until
  the next hit. So “10 consecutive” means **10 consecutive YOLO passes** with no activity
  (not necessarily 10 video frames, because YOLO may not run every frame).

**Flow**  
Each frame: ``should_run_yolo`` → if true, run YOLO → ``observe(has_activity)``.
If YOLO was skipped, do not call ``observe``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YoloStrideGateConfig:
    """Tuning bundle for :class:`YoloStrideGate` (usually built from ``settings.py`` / env)."""

    coarse_stride: int = 8
    """When idle: run YOLO once per this many frame indices (typical 5–10)."""

    dense_stride: int = 2
    """While dense mode is on: also run YOLO every this many frames (2 = every other frame)."""

    dense_idle_miss_streak: int = 10
    """Exit dense mode after this many **consecutive** YOLO runs with no person/vehicle."""


class YoloStrideGate:
    """Coarse sampling by default; densify after a person/vehicle hit until idle miss streak."""

    def __init__(self, cfg: YoloStrideGateConfig) -> None:
        self._c = cfg
        self._dense_active = False
        self._consecutive_misses = 0

    def should_run_yolo(self, frame_idx: int) -> bool:
        """True if this frame index should run YOLO (and then LP/OCR if the pipeline does so)."""
        coarse = (frame_idx % max(1, self._c.coarse_stride)) == 0
        dense = self._dense_active and ((frame_idx % max(1, self._c.dense_stride)) == 0)
        return coarse or dense

    def observe(self, _frame_idx: int, has_activity: bool) -> None:
        """Call only on frames where YOLO actually ran; pass whether person/vehicle were found."""
        streak = max(1, self._c.dense_idle_miss_streak)
        if has_activity:
            self._dense_active = True
            self._consecutive_misses = 0
            return
        if not self._dense_active:
            self._consecutive_misses = 0
            return
        self._consecutive_misses += 1
        if self._consecutive_misses >= streak:
            self._dense_active = False
            self._consecutive_misses = 0
