"""Cumulative counters for scene / LP Ultralytics ``model()`` invocations (``.pt`` or TRT)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UltralyticsCallStats:
    """One detector's totals over a pipeline run."""

    input_units: int
    """Frames (scene YOLO) or vehicle crops (LP) passed into ``detect`` / ``detect_plates``."""

    batch_launches: int
    """Number of ``model()`` forwards: one per ``detect`` call for ``.pt``; one per TRT chunk for ``.engine``."""

    padded_slots: int
    """Dummy images actually appended for **static** TensorRT (``*_TRT_DYNAMIC=False``): ``Σ max(0, B - n)``."""

    max_batch_slack: int = 0
    """**Dynamic** TensorRT only: ``Σ max(0, B - n)`` per launch — unused *capacity* vs engine max batch ``B``.

    Not dummy inferences (Ultralytics still sends ``n`` real tensors). Shown so gated 1-frame calls are not
    misread as thousands of "padded" frames.
    """


def empty_ultralytics_call_stats() -> UltralyticsCallStats:
    return UltralyticsCallStats(0, 0, 0, 0)


def trt_engine_chunk_plan(total_inputs: int, batch_size: int, *, dynamic: bool) -> UltralyticsCallStats:
    """Dry-run the TensorRT chunk loop (must stay aligned with ``YoloDetector`` / ``LpDetector``)."""
    B = max(1, int(batch_size))
    n = max(0, int(total_inputs))
    launches = 0
    padded = 0
    slack = 0
    for start in range(0, n, B):
        valid = min(B, n - start)
        launches += 1
        short = max(0, B - valid)
        if dynamic:
            slack += short
        else:
            padded += short
    return UltralyticsCallStats(n, launches, padded, slack)
