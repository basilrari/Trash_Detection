#!/usr/bin/env python3
"""Export RF-DETR ``trash.pth`` + ``cigarette.pth`` to ONNX for fast inference.

Run from ``worker-python/``::

  pip install "rfdetr[onnx]"
  python scripts/export_rfdetr_heads.py \\
    --trash weights/trash.pth \\
    --cigarette weights/cigarette.pth \\
    --output-dir exports/rfdetr_onnx \\
    --shape 640 640 \\
    --batch-size 8

Then build a TensorRT engine (FP16) on the **same GPU** you deploy on::

  trtexec --onnx=exports/rfdetr_onnx/trash/inference_model.onnx \\
    --saveEngine=exports/rfdetr_onnx/trash/inference_model.engine \\
    --fp16

Or use Roboflow helper (writes ``inference_model.engine`` next to the ONNX by default)::

  python -c \"from argparse import Namespace; from rfdetr.export.tensorrt import trtexec; \\
    trtexec('exports/rfdetr_onnx/trash/inference_model.onnx', Namespace(verbose=True, profile=False, dry_run=False))\"

Pipeline (ONNX Runtime, static batch ``B`` from export)::

  export RF_DETR_BACKEND=onnx
  export RF_DETR_TRASH_ONNX=exports/rfdetr_onnx/trash/inference_model.onnx
  export RF_DETR_CIGARETTE_ONNX=exports/rfdetr_onnx/cigarette/inference_model.onnx
  pip install onnxruntime-gpu
  python worker.py inputs/clip.mp4 -o outputs/out.mp4

``--no-optimize`` skips ``optimize_for_inference`` entirely before export (recommended if export fails).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKER_ROOT = Path(__file__).resolve().parents[1]
if str(WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKER_ROOT))


def _export_one(
    weights: Path,
    out_subdir: Path,
    *,
    shape: tuple[int, int],
    batch_size: int,
    run_optimize: bool,
    compile_export: bool,
    export_dtype: str,
) -> Path:
    import torch

    from models.trash_detector import _build_rfdetr

    model = _build_rfdetr(str(weights), run_optimize=False)
    out_subdir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if export_dtype == "fp16" else torch.float32
    opt = getattr(model, "optimize_for_inference", None)
    if run_optimize and callable(opt):
        try:
            opt(compile=compile_export, dtype=dtype, batch_size=batch_size)
        except TypeError:
            try:
                opt(compile=False, dtype=torch.float32, batch_size=batch_size)
            except TypeError:
                opt(compile=False, batch_size=1)
    h, w = shape
    model.export(
        output_dir=str(out_subdir),
        format="onnx",
        shape=(h, w),
        batch_size=batch_size,
        verbose=True,
    )
    onnx_path = out_subdir / "inference_model.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Export did not produce {onnx_path}")
    return onnx_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trash", type=Path, required=True, help="Path to trash.pth")
    parser.add_argument("--cigarette", type=Path, required=True, help="Path to cigarette.pth")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKER_ROOT / "exports" / "rfdetr_onnx",
        help="Parent directory; creates trash/ and cigarette/ subfolders",
    )
    parser.add_argument("--shape", type=int, nargs=2, default=(640, 640), metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip optimize_for_inference entirely before export",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Pass compile=True into optimize_for_inference when supported",
    )
    parser.add_argument(
        "--export-dtype",
        choices=("fp32", "fp16"),
        default="fp16",
        help="dtype passed to optimize_for_inference when supported",
    )
    args = parser.parse_args()

    shape = (int(args.shape[0]), int(args.shape[1]))
    out_root = args.output_dir.resolve()
    run_optimize = not args.no_optimize

    tp = _export_one(
        args.trash,
        out_root / "trash",
        shape=shape,
        batch_size=args.batch_size,
        run_optimize=run_optimize,
        compile_export=args.compile,
        export_dtype=args.export_dtype,
    )
    print("Trash ONNX:", tp)
    cp = _export_one(
        args.cigarette,
        out_root / "cigarette",
        shape=shape,
        batch_size=args.batch_size,
        run_optimize=run_optimize,
        compile_export=args.compile,
        export_dtype=args.export_dtype,
    )
    print("Cigarette ONNX:", cp)
    print("\nExample trtexec (run on target GPU):")
    for name in ("trash", "cigarette"):
        ox = out_root / name / "inference_model.onnx"
        eng = out_root / name / "inference_model.engine"
        print(f"  trtexec --onnx={ox} --saveEngine={eng} --fp16")
    print("\nExample worker env:")
    print("  export RF_DETR_BACKEND=onnx")
    print(f"  export RF_DETR_TRASH_ONNX={out_root / 'trash' / 'inference_model.onnx'}")
    print(f"  export RF_DETR_CIGARETTE_ONNX={out_root / 'cigarette' / 'inference_model.onnx'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
