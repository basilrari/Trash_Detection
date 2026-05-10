"""Cross-frame LP crop batching with emit-aware flushing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from models.types import FrameData


@dataclass(frozen=True)
class LpQueuedCrop:
    frame_idx: int
    tid: int
    vx1: int
    vy1: int
    vx2: int
    vy2: int
    frame_w: int
    frame_h: int
    vehicle_crop: np.ndarray


class LpBatchCoordinator:
    """Batch LP inference across frames; flush by batch size, latency, or emit boundary."""

    def __init__(
        self,
        *,
        lp_detector: Any,
        ocr: Any,
        cache: Any,
        times: Any,
        max_crops: int,
        max_latency_frames: int,
        enabled: bool,
    ) -> None:
        self._lp = lp_detector
        self._ocr = ocr
        self._cache = cache
        self._times = times
        self._max_crops = max(1, int(max_crops))
        self._max_lat = max(0, int(max_latency_frames))
        self._enabled = bool(enabled)
        self._q: list[LpQueuedCrop] = []
        self.lp_queue_flushes: int = 0
        self.lp_latency_flushes: int = 0
        self.lp_emit_flushes: int = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enqueue_vehicle_crop(
        self,
        *,
        frame_idx: int,
        tid: int,
        vx1: int,
        vy1: int,
        vx2: int,
        vy2: int,
        frame_w: int,
        frame_h: int,
        vehicle_crop: np.ndarray,
    ) -> None:
        if not self._enabled:
            return
        self._q.append(
            LpQueuedCrop(
                frame_idx=int(frame_idx),
                tid=int(tid),
                vx1=int(vx1),
                vy1=int(vy1),
                vx2=int(vx2),
                vy2=int(vy2),
                frame_w=int(frame_w),
                frame_h=int(frame_h),
                vehicle_crop=np.ascontiguousarray(vehicle_crop),
            )
        )

    def after_enqueue(self, frame_idx: int) -> None:
        """Flush full LP batches or latency windows only (no emit-boundary drain)."""
        if not self._enabled:
            return
        while len(self._q) >= self._max_crops:
            self._flush_first_n(self._max_crops)
        if self._max_lat > 0 and self._q:
            oldest = self._q[0].frame_idx
            if frame_idx - oldest >= self._max_lat:
                self.lp_latency_flushes += 1
                self._flush_first_n(min(len(self._q), self._max_crops))

    def flush_until_frame_ready(self, max_frame: int) -> None:
        if not self._enabled:
            return
        while self._q and self._q[0].frame_idx <= max_frame:
            self.lp_emit_flushes += 1
            n_take = 0
            for j in range(len(self._q)):
                if self._q[j].frame_idx > max_frame:
                    break
                n_take += 1
                if n_take >= self._max_crops:
                    break
            if n_take <= 0:
                break
            self._flush_first_n(n_take)

    def eof_flush(self) -> None:
        if not self._enabled:
            return
        while self._q:
            n = min(len(self._q), self._max_crops)
            self._flush_first_n(n)

    def _flush_first_n(self, n: int) -> None:
        if n <= 0 or not self._q:
            return
        chunk = self._q[:n]
        del self._q[:n]
        self.lp_queue_flushes += 1

        fd_list = [
            FrameData(index=k, timestamp=0.0, image=j.vehicle_crop) for k, j in enumerate(chunk)
        ]
        t0 = time.perf_counter()
        plates_per = self._lp.detect_plates(fd_list)
        self._times.lp_sec += time.perf_counter() - t0

        self._cache.apply_lp_chunk_results(
            chunk,
            plates_per,
            lp_detector=self._lp,
            ocr=self._ocr,
            times=self._times,
        )
