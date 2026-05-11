#!/usr/bin/env python3
"""
Batch every video under ``inputs/Cases`` with models loaded once (mirrors paths under ``outputs/Cases``).

Run from ``worker-python/`` so imports and paths in ``settings.py`` resolve::

  cd worker-python
  python scripts/batch_cases.py
  python scripts/batch_cases.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# ``python scripts/batch_cases.py`` adds ``scripts/`` to sys.path, not worker-python — insert repo root.
_WORKER_PYTHON_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKER_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKER_PYTHON_ROOT))

from rich.console import Console

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

MANIFEST_FIELDS = [
    "input_path",
    "output_path",
    "success",
    "error",
    "duration_sec",
    "fps",
    "width",
    "height",
    "total_frames",
    "wall_sec",
    "inference_sec",
    "encode_io_sec",
    "models_init_sec",
]

console = Console()


def discover_videos(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            out.append(p)
    return out


def mirrored_output_path(video: Path, input_root: Path, output_root: Path) -> Path:
    rel = video.relative_to(input_root.resolve())
    return output_root / rel.parent / f"{rel.stem}_annotated{rel.suffix}"


def append_manifest_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pipeline on every video under inputs/Cases (single model load).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("inputs/Cases"),
        help="Directory tree to scan for videos (default: inputs/Cases).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/Cases"),
        help="Root for mirrored outputs (default: outputs/Cases).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/Cases_batch_manifest.csv"),
        help="CSV manifest path (default: outputs/Cases_batch_manifest.csv).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned input→output pairs only.",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    input_root = (cwd / args.input_dir).resolve()
    output_root = (cwd / args.output_dir).resolve()

    videos = discover_videos(input_root)
    if not videos:
        console.print(f"[red]No videos found under[/] {input_root}")
        sys.exit(1)

    jobs: list[tuple[Path, Path]] = []
    for v in videos:
        jobs.append((v, mirrored_output_path(v, input_root, output_root)))

    console.print(f"[bold]Found {len(jobs)} video(s)[/] under {input_root}")
    if args.dry_run:
        for inp, outp in jobs:
            console.print(f"  {inp}  →  {outp}")
        return

    from pipelines.test_pipeline import load_pipeline_models, run_pipeline_video

    manifest_path = (cwd / args.manifest).resolve()
    batch_wall_start = time.perf_counter()
    bundle = load_pipeline_models()
    total_inference_sec = 0.0
    total_source_duration_sec = 0.0
    n_ok = 0
    n_fail = 0
    try:
        for i, (inp, outp) in enumerate(jobs):
            console.print(f"[bold cyan][{i + 1}/{len(jobs)}][/] {inp.name}")
            outp.parent.mkdir(parents=True, exist_ok=True)
            rec = run_pipeline_video(
                bundle,
                str(inp),
                str(outp),
                per_video_times_init_sec=0.0,
                models_init_sec=bundle.init_sec,
                abort_on_error=False,
            )
            if rec.success:
                n_ok += 1
                console.print(
                    f"  [green]ok[/] wall={rec.wall_sec:.2f}s  "
                    f"inference={rec.inference_sec:.2f}s  encode/io={rec.encode_io_sec:.2f}s  → {outp}"
                )
            else:
                n_fail += 1
                console.print(f"  [red]failed[/] {rec.error}")

            total_inference_sec += rec.inference_sec
            total_source_duration_sec += rec.duration_sec

            row = {
                "input_path": str(inp),
                "output_path": str(outp),
                "success": rec.success,
                "error": rec.error or "",
                "duration_sec": f"{rec.duration_sec:.6f}",
                "fps": f"{rec.fps:.6f}",
                "width": rec.width,
                "height": rec.height,
                "total_frames": rec.total_frames,
                "wall_sec": f"{rec.wall_sec:.6f}",
                "inference_sec": f"{rec.inference_sec:.6f}",
                "encode_io_sec": f"{rec.encode_io_sec:.6f}",
                "models_init_sec": f"{rec.models_init_sec:.6f}",
            }
            append_manifest_row(manifest_path, row)
    finally:
        bundle.cleanup()

    batch_wall_sec = time.perf_counter() - batch_wall_start

    console.print()
    console.print("[bold]Totals[/]")
    console.print(f"  Videos:                  {len(jobs)}  ([green]{n_ok} ok[/], [red]{n_fail} failed[/])")
    console.print(
        "  Total time taken:        "
        f"[bold]{batch_wall_sec:.2f} s[/]  [dim](model load through last video + teardown)[/]"
    )
    console.print(
        "  Total inference time:    "
        f"[bold]{total_inference_sec:.2f} s[/]  [dim](sum of per-file inference)[/]"
    )
    console.print(
        "  Total source video time: "
        f"[bold]{total_source_duration_sec:.2f} s[/]  [dim](sum of input durations from metadata)[/]"
    )
    console.print(f"[bold]Manifest:[/] {manifest_path}")


if __name__ == "__main__":
    main()
