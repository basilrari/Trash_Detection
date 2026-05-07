#!/usr/bin/env python3
"""
Local video pipeline: YOLO → license plate detector → OCR → annotated MP4.

Examples (paths relative to worker-python/):

  python worker.py
  python worker.py inputs/clip.mp4
  python worker.py inputs/clip.mp4 -o outputs/custom.mp4

Gating (optional, ``GATE_MODE=yolo``):
  When the gate is on, we do not run the full YOLO+LP+OCR stack on every frame.
  YOLO is used twice: (1) as the detector, (2) as the signal for when to run more
  often — see ``core/yolo_stride_gate.py`` and env vars below.

  # YOLO-only stride gate: coarse when idle, denser after people/vehicles appear
  GATE_MODE=yolo python worker.py inputs/clip.mp4

  # Same, with inline tuning (see Readme "Gating" section for meanings)
  GATE_MODE=yolo YOLO_COARSE_STRIDE=10 YOLO_DENSE_STRIDE=2 YOLO_DENSE_WINDOW_SEC=5 \\
    python worker.py inputs/clip.mp4 -o outputs/out.mp4
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def default_output_path(video_path: str, outputs_dir: str) -> str:
    p = Path(video_path)
    name = f"{p.stem}_annotated{p.suffix}"
    return str(Path(outputs_dir) / name)


def main() -> None:
    # -------------------------------------------------------------------------
    # Argparse must run before importing ``settings`` so ``os.environ`` overrides
    # (e.g. --gate yolo) are visible when defaults are read from the environment.
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO + license-plate detector + OCR on a video file.\n\n"
            "GATE_MODE:\n"
            "  off  — Original behavior: YOLO on every frame inside each time chunk "
            "(CHUNK_SECONDS in settings.py).\n"
            "  yolo — YOLO-only gating: run YOLO coarsely when the scene looks idle, "
            "and more often for a short window after a person or vehicle is seen. "
            "Tuning: YOLO_COARSE_STRIDE, YOLO_DENSE_STRIDE, YOLO_DENSE_WINDOW_SEC "
            "(env or settings.py). See Readme.md § Gating."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Input video path (default: VIDEO_PATH in settings.py when omitted).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Annotated output path (default: OUTPUT_VIDEO in settings, or <stem>_annotated in outputs/).",
    )
    parser.add_argument(
        "--gate",
        choices=["off", "yolo"],
        default=None,
        help=(
            "Override GATE_MODE for this run only: "
            "'off' = full-rate YOLO within each chunk; "
            "'yolo' = coarse/dense YOLO stride gate (no MOG2). "
            "Default: value from settings.py / environment."
        ),
    )
    parser.add_argument(
        "--yolo-coarse-stride",
        type=int,
        default=None,
        metavar="N",
        help=(
            "When GATE_MODE=yolo: run YOLO at least every N frames while no person/vehicle "
            "was seen on the last gated runs (typical 5–10). Env: YOLO_COARSE_STRIDE."
        ),
    )
    parser.add_argument(
        "--yolo-dense-stride",
        type=int,
        default=None,
        metavar="N",
        help=(
            "When GATE_MODE=yolo: after activity, run YOLO every N frames while the dense "
            "window is open (2 = every other frame). Env: YOLO_DENSE_STRIDE."
        ),
    )
    parser.add_argument(
        "--yolo-dense-window-sec",
        type=float,
        default=None,
        metavar="SEC",
        help=(
            "When GATE_MODE=yolo: after a person/vehicle hit, keep the denser schedule for "
            "this many seconds of video (converted to frames via FPS). Env: YOLO_DENSE_WINDOW_SEC."
        ),
    )

    args = parser.parse_args()

    if args.gate is not None:
        os.environ["GATE_MODE"] = args.gate
    if args.yolo_coarse_stride is not None:
        os.environ["YOLO_COARSE_STRIDE"] = str(args.yolo_coarse_stride)
    if args.yolo_dense_stride is not None:
        os.environ["YOLO_DENSE_STRIDE"] = str(args.yolo_dense_stride)
    if args.yolo_dense_window_sec is not None:
        os.environ["YOLO_DENSE_WINDOW_SEC"] = str(args.yolo_dense_window_sec)

    from settings import OUTPUT_VIDEO, OUTPUTS_DIR, VIDEO_PATH
    from pipelines.test_pipeline import run_pipeline

    video_path = args.video or VIDEO_PATH
    if not os.path.isfile(video_path):
        raise SystemExit(f"Video not found: {video_path}")

    out_path = args.output
    if out_path is None:
        out_path = OUTPUT_VIDEO if args.video is None else default_output_path(video_path, OUTPUTS_DIR)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    run_pipeline(video_path, out_path)


if __name__ == "__main__":
    main()
