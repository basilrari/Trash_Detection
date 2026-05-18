"""Synthetic checks for peeing bbox stillness (no YOLO / TensorRT load).

Run from ``worker-python/``:

  python scripts/smoke_peeing_stillness_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.peeing_stillness import bbox_is_still


def main() -> None:
    base = (100.0, 100.0, 200.0, 300.0)
    tiny = (101.0, 101.0, 199.0, 299.0)
    shifted = (150.0, 100.0, 250.0, 300.0)
    kw = dict(
        min_iou=0.65,
        max_center_motion_norm=0.035,
        max_size_change=0.12,
    )
    assert bbox_is_still(base, tiny, **kw), "tiny jitter should count as still"
    assert not bbox_is_still(base, shifted, **kw), "large translation should count as motion"


if __name__ == "__main__":
    main()
    print("smoke_peeing_stillness_gate: OK", file=sys.stderr)
