"""Tests for emit-aware ``LpBatchCoordinator``."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.types import LicensePlate  # noqa: E402
from pipelines.lp_batch_coordinator import LpBatchCoordinator  # noqa: E402


class _Times:
    lp_sec: float = 0.0


class LpBatchCoordinatorEmitTest(unittest.TestCase):
    def test_cross_frame_batch_fill_without_per_frame_flush(self) -> None:
        """Queued crops from two frames can share one LP launch before emit drains."""
        times = _Times()
        cache = MagicMock()
        cache.apply_lp_chunk_results = MagicMock()
        det = MagicMock()
        det.detect_plates = MagicMock(
            side_effect=[
                [[LicensePlate(bbox=(0, 0, 1, 1), text="", confidence=0.9)]],
                [[LicensePlate(bbox=(0, 0, 1, 1), text="", confidence=0.91)]],
            ]
        )
        coord = LpBatchCoordinator(
            lp_detector=det,
            ocr=MagicMock(),
            cache=cache,
            times=times,
            max_crops=2,
            max_latency_frames=0,
            enabled=True,
        )
        z = [[[0, 0, 0] for _ in range(10)] for _ in range(10)]
        coord.enqueue_vehicle_crop(
            frame_idx=0,
            tid=0,
            vx1=0,
            vy1=0,
            vx2=5,
            vy2=5,
            frame_w=100,
            frame_h=100,
            vehicle_crop=z,
        )
        coord.enqueue_vehicle_crop(
            frame_idx=1,
            tid=1,
            vx1=0,
            vy1=0,
            vx2=5,
            vy2=5,
            frame_w=100,
            frame_h=100,
            vehicle_crop=z,
        )
        coord.after_enqueue(1)
        self.assertEqual(len(coord._q), 2)  # noqa: SLF001
        det.detect_plates.assert_not_called()

        coord.flush_until_frame_ready(0)
        det.detect_plates.assert_called_once()
        self.assertEqual(len(coord._q), 1)  # noqa: SLF001 — frame 1 still pending

        coord.flush_until_frame_ready(1)
        self.assertEqual(det.detect_plates.call_count, 2)
        self.assertEqual(len(coord._q), 0)  # noqa: SLF001

    def test_emit_barrier_skips_when_queue_empty(self) -> None:
        times = _Times()
        cache = MagicMock()
        det = MagicMock()
        coord = LpBatchCoordinator(
            lp_detector=det,
            ocr=MagicMock(),
            cache=cache,
            times=times,
            max_crops=4,
            max_latency_frames=0,
            enabled=True,
        )
        coord.flush_until_frame_ready(0)
        self.assertEqual(coord.lp_emit_flushes, 0)


if __name__ == "__main__":
    unittest.main()
