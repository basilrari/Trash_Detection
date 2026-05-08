#!/usr/bin/env python3
"""Run ``worker.py`` on every video in a folder (annotated MP4 per file).

From ``worker-python/``::

  python scripts/batch_worker_videos.py
  python scripts/batch_worker_videos.py -i inputs -o outputs
  python scripts/batch_worker_videos.py -i inputs -o outputs --gate off

Uses the same pipeline as a manual ``python worker.py <video> -o <out>`` run.
Filenames with spaces are handled via ``subprocess`` argument lists.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

WORKER_ROOT = Path(__file__).resolve().parents[1]
WORKER_PY = WORKER_ROOT / "worker.py"

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def _iter_videos(input_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in VIDEO_SUFFIXES:
            out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=WORKER_ROOT / "inputs",
        help=f"Folder of videos (default: {WORKER_ROOT / 'inputs'})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=WORKER_ROOT / "outputs",
        help=f"Annotated MP4s written here (default: {WORKER_ROOT / 'outputs'})",
    )
    parser.add_argument(
        "--gate",
        choices=("off", "yolo"),
        default=None,
        help="Passed to worker.py (default: worker/settings default, usually yolo).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a clip if the target annotated file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not run worker.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = _iter_videos(input_dir)
    if not videos:
        print(f"No video files ({', '.join(VIDEO_SUFFIXES)}) in {input_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(videos)} video(s) under {input_dir}")
    failed: list[tuple[Path, int]] = []
    skipped = 0

    for vp in videos:
        out_name = f"{vp.stem}_annotated{vp.suffix.lower()}"
        out_path = output_dir / out_name
        if args.skip_existing and out_path.is_file():
            print(f"[skip] exists: {out_path.name}")
            skipped += 1
            continue

        cmd = [sys.executable, str(WORKER_PY), str(vp), "-o", str(out_path)]
        if args.gate is not None:
            cmd.extend(["--gate", args.gate])

        print(f"\n=== {vp.name} -> {out_path.name} ===", flush=True)
        if args.dry_run:
            print(" ", subprocess.list2cmdline(cmd))
            continue

        env = os.environ.copy()
        r = subprocess.run(cmd, cwd=str(WORKER_ROOT), env=env)
        if r.returncode != 0:
            failed.append((vp, r.returncode))

    if args.dry_run:
        return 0

    print(f"\nDone. ok={len(videos) - skipped - len(failed)} skipped={skipped} failed={len(failed)}")
    for p, code in failed:
        print(f"  FAILED ({code}): {p.name}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
