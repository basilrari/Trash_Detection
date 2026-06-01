#!/usr/bin/env python3
"""
Minimal peeing pipeline — **re-export** of :mod:`pipelines.peeing_pipeline`.

Loads and runs only what peeing needs with the same logic as the full worker:

- **Scene YOLO** (stride + micro-batch, carried boxes between samples)
- **PeeingDetector** (YOLO pose TRT, motorcycle gate, stillness, temporal buckets)

No RF-DETR, no LP detector, no OCR. For programmatic use::

    from pipelines.minimal_peeing_pipeline import run_peeing_pipeline, load_peeing_only_models

Implementation lives in ``peeing_pipeline.py``; this module exists so imports can say
``minimal_peeing_pipeline`` without duplicating code.
"""

from __future__ import annotations

from pipelines.peeing_pipeline import (
    PeeingOnlyModelBundle,
    PeeingOnlyStepTimes,
    PeeingOnlyVideoRecord,
    load_peeing_only_models,
    run_peeing_pipeline,
    run_peeing_pipeline_video,
)

__all__ = [
    "PeeingOnlyModelBundle",
    "PeeingOnlyStepTimes",
    "PeeingOnlyVideoRecord",
    "load_peeing_only_models",
    "run_peeing_pipeline",
    "run_peeing_pipeline_video",
]
