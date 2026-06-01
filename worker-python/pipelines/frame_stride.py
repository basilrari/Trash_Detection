"""Uniform scene-YOLO stride from reported FPS and ``settings.py``."""

from __future__ import annotations

from settings import (
    FRAME_SAMPLE_STRIDE_OVERRIDE,
    INPUT_VIDEO_FPS_MAX,
    INPUT_VIDEO_FPS_MIN,
    SCENE_YOLO_TARGET_FRAMES_PER_SECOND,
)


def _resolve_frame_sample_stride(fps: float) -> tuple[int, str]:
    """Pick decoded-frame stride: optional fixed override, else ~``fps/target`` frames sampled per second."""
    if FRAME_SAMPLE_STRIDE_OVERRIDE is not None:
        n = max(1, int(FRAME_SAMPLE_STRIDE_OVERRIDE))
        return n, f"fixed ``FRAME_SAMPLE_STRIDE_OVERRIDE={n}``"
    target = float(SCENE_YOLO_TARGET_FRAMES_PER_SECOND)
    if target <= 0:
        target = 5.0
    lo = float(INPUT_VIDEO_FPS_MIN)
    hi = float(INPUT_VIDEO_FPS_MAX)
    fps_clamped = min(max(float(fps), lo), hi)
    n = max(1, int(round(fps_clamped / target)))
    approx_per_sec = fps_clamped / float(n)
    return n, (
        f"automatic: reported FPS={fps:.3f}; clamp [{lo:.0f},{hi:.0f}] → {fps_clamped:.2f}; "
        f"``SCENE_YOLO_TARGET_FRAMES_PER_SECOND={target:g}`` → stride={n} "
        f"(~{approx_per_sec:.2f} scene-YOLO frames/s of video)"
    )
