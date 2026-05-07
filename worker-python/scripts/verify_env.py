#!/usr/bin/env python3
"""Smoke-test imports, CUDA, YOLO weights (optional), and PaddleOCR."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# worker-python as cwd for imports
WORKER_ROOT = Path(__file__).resolve().parents[1]
if str(WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKER_ROOT))

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def main() -> int:
    print("Python:", sys.version.split()[0])

    import cv2
    import numpy as np
    import torch

    print("OpenCV:", cv2.__version__)
    print("PyTorch:", torch.__version__, "cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print("CUDA device:", name, f"capability sm_{cap[0]}{cap[1]}")
        if cap[0] >= 12:
            print(
                "NOTE: This GPU may need a PyTorch build with sm_120+ support, or use an "
                "older-GPU index, e.g. CUDA_VISIBLE_DEVICES=1 for a second card."
            )

    import paddle

    print("Paddle:", paddle.__version__, "compiled_with_cuda:", paddle.device.is_compiled_with_cuda())

    from ultralytics import YOLO

    print("Ultralytics OK")

    weights_yolo = WORKER_ROOT / "weights" / "yolo11x.pt"
    weights_lp = WORKER_ROOT / "weights" / "bestlicense.pt"
    if weights_yolo.is_file():
        m = YOLO(str(weights_yolo))
        print("YOLO weights load OK:", weights_yolo)
        del m
    else:
        print("SKIP YOLO weights (optional):", weights_yolo, "not found")

    if weights_lp.is_file():
        m2 = YOLO(str(weights_lp))
        print("LP weights load OK:", weights_lp)
        del m2
    else:
        print("SKIP LP weights (optional):", weights_lp, "not found")

    from models.ocr import Ocr

    ocr = Ocr()
    blank = np.zeros((48, 160, 3), dtype=np.uint8)
    cv2.putText(blank, "X0", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    out = ocr.recognize([blank])
    print("Ocr.recognize smoke OK:", out)

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
