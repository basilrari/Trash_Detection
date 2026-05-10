"""Smoke test: cross-frame prefetch list alignment and no duplicate YOLO pose forwards.

Run from ``worker-python/``:

  python scripts/smoke_peeing_pose_prefetch.py

Uses ``unittest.mock`` to avoid loading real pose weights. Creates a minimal ``.engine`` file
because the detector requires a TensorRT engine path.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# worker-python root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.peeing_detector import PeeingDetector
from models.types import Detection


def main() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # One person box tall enough for min crop / margin.
    dets = [
        Detection(
            bbox=(200.0, 50.0, 400.0, 400.0),
            label="person",
            confidence=0.95,
        )
    ]

    infer_calls: list[int] = []

    def fake_infer(_self, crops):
        infer_calls.append(len(crops))
        return [True] * len(crops)

    with tempfile.NamedTemporaryFile(suffix=".engine", delete=False) as tmp:
        tmp.write(b"\0")
        engine_path = tmp.name

    try:
        with patch("ultralytics.YOLO", return_value=MagicMock()):
            peeing = PeeingDetector(
                pose_backend="yolo",
                yolo_pose_model=engine_path,
                yolo_pose_batch_size=8,
            )

        with patch.object(PeeingDetector, "_infer_yolo_pose_batch", fake_infer):
            hits_map = peeing.prefetch_yolo_pose_hits_for_window(
                [(0, frame, dets)],
                yolo_conf=0.25,
            )
            row = hits_map[0]
            assert len(row) == 1 and row[0] is True, row
            assert infer_calls == [1], infer_calls

            infer_calls.clear()
            st = peeing.update(
                frame,
                dets,
                run_yolo=True,
                yolo_conf=0.25,
                timestamp_sec=0.0,
                precomputed_yolo_pose_hits=row,
            )
            assert infer_calls == [], (
                f"expected no pose infer when precomputed, got {infer_calls}"
            )
            assert st.sampled is True
    finally:
        Path(engine_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
    print("smoke_peeing_pose_prefetch: OK", file=sys.stderr)
