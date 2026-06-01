#!/usr/bin/env python3
"""
Batch minimal peeing pipeline on every video under ``inputs/long_test`` (models loaded once).

Mirrors directory layout under ``outputs/long_test_peeing`` with ``<stem>_peeing_annotated<suffix>``.
Writes two CSVs: a per-video summary (timing + metadata) and per-interval peeing events.

Run from ``worker-python/``::

  cd worker-python
  python scripts/batch_long_test_peeing.py
  python scripts/batch_long_test_peeing.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

_WORKER_PYTHON_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKER_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKER_PYTHON_ROOT))

from rich.console import Console

from pipelines.peeing_pipeline import PeeingOnlyVideoRecord, load_peeing_only_models, run_peeing_pipeline_video

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

SUMMARY_FIELDS = [
    "input_path",
    "output_path",
    "success",
    "error",
    "width",
    "height",
    "fps",
    "total_frames",
    "duration_sec",
    "video_duration_str",
    "wall_sec",
    "wall_duration_str",
    "completed_in_min_sec",
    "models_init_sec",
    "other_sec",
    "yolo_sec",
    "peeing_sec",
    "annotate_draw_sec",
    "video_write_sec",
    "inference_sec",
    "encode_io_sec",
    "yolo_input_frames",
    "yolo_batch_launches",
    "yolo_padded_slots",
    "yolo_max_batch_slack",
    "peeing_pose_input_crops",
    "peeing_pose_batch_launches",
    "peeing_pose_padded_slots",
    "peeing_pose_max_batch_slack",
    "peeing_pose_prefetch_windows",
    "peeing_pose_prefetch_frames",
    "peeing_pose_prefetch_crops",
    "peeing_pose_prefetch_unused_hits",
    "peeing_detected",
    "peeing_event_count",
]

EVENT_FIELDS = [
    "input_path",
    "output_path",
    "event_index",
    "start_frame",
    "end_frame",
    "start_sec",
    "end_sec",
    "duration_sec",
    "start_time_str",
    "end_time_str",
    "duration_str",
    "n_confirmed_bboxes",
    "bboxes_json",
    "open_interval",
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


def mirrored_peeing_output_path(video: Path, input_root: Path, output_root: Path) -> Path:
    rel = video.relative_to(input_root.resolve())
    return output_root / rel.parent / f"{rel.stem}_peeing_annotated{rel.suffix}"


def _format_video_duration(duration_sec: float) -> str:
    if duration_sec < 0:
        duration_sec = 0.0
    m = int(duration_sec // 60)
    s = duration_sec % 60.0
    if m == 0:
        return f"{s:.2f}s"
    return f"{m}m {s:.2f}s"


def _format_wall_min_sec(wall_sec: float) -> str:
    """e.g. ``3m 12.50s`` for processing wall time."""
    if wall_sec < 0:
        wall_sec = 0.0
    m = int(wall_sec // 60)
    s = wall_sec % 60.0
    if m == 0:
        return f"{s:.2f}s"
    return f"{m}m {s:.2f}s"


def summary_row_dict(rec: PeeingOnlyVideoRecord, *, models_init_sec: float) -> dict[str, object]:
    out: dict[str, object] = {k: "" for k in SUMMARY_FIELDS}
    out["input_path"] = rec.input_path
    out["output_path"] = rec.output_path
    out["success"] = rec.success
    out["error"] = rec.error or ""
    out["width"] = rec.width
    out["height"] = rec.height
    out["wall_sec"] = f"{rec.wall_sec:.6f}"
    out["wall_duration_str"] = _format_video_duration(rec.wall_sec)
    out["completed_in_min_sec"] = _format_wall_min_sec(rec.wall_sec)
    out["models_init_sec"] = f"{models_init_sec:.6f}"
    out["peeing_event_count"] = len(rec.events)
    out["peeing_detected"] = len(rec.events) > 0

    if rec.success:
        out["fps"] = f"{rec.fps:.6f}"
        out["total_frames"] = rec.total_frames
        out["duration_sec"] = f"{rec.duration_sec:.6f}"
        out["video_duration_str"] = _format_video_duration(rec.duration_sec)
        out["inference_sec"] = f"{rec.inference_sec:.6f}"
        out["encode_io_sec"] = f"{rec.encode_io_sec:.6f}"

    t = rec.times
    if t is not None:
        out["other_sec"] = f"{t.other_sec:.6f}"
        out["yolo_sec"] = f"{t.yolo_sec:.6f}"
        out["peeing_sec"] = f"{t.peeing_sec:.6f}"
        out["annotate_draw_sec"] = f"{t.annotate_draw_sec:.6f}"
        out["video_write_sec"] = f"{t.video_write_sec:.6f}"
        out["yolo_input_frames"] = t.yolo_input_frames
        out["yolo_batch_launches"] = t.yolo_batch_launches
        out["yolo_padded_slots"] = t.yolo_padded_slots
        out["yolo_max_batch_slack"] = t.yolo_max_batch_slack
        out["peeing_pose_input_crops"] = t.peeing_pose_input_crops
        out["peeing_pose_batch_launches"] = t.peeing_pose_batch_launches
        out["peeing_pose_padded_slots"] = t.peeing_pose_padded_slots
        out["peeing_pose_max_batch_slack"] = t.peeing_pose_max_batch_slack
        out["peeing_pose_prefetch_windows"] = t.peeing_pose_prefetch_windows
        out["peeing_pose_prefetch_frames"] = t.peeing_pose_prefetch_frames
        out["peeing_pose_prefetch_crops"] = t.peeing_pose_prefetch_crops
        out["peeing_pose_prefetch_unused_hits"] = t.peeing_pose_prefetch_unused_hits

    return {k: out[k] for k in SUMMARY_FIELDS}


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run minimal peeing pipeline on every video under inputs/long_test (single model load).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("inputs/Long_Test"),
        help="Directory tree to scan for videos (default: inputs/long_test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/long_test_peeing"),
        help="Root for mirrored outputs (default: outputs/long_test_peeing).",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/long_test_peeing_summary.csv"),
        help="Per-video timing summary CSV (default: outputs/long_test_peeing_summary.csv).",
    )
    parser.add_argument(
        "--events-csv",
        type=Path,
        default=Path("outputs/long_test_peeing_events.csv"),
        help="Per-interval peeing events CSV (default: outputs/long_test_peeing_events.csv).",
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
        jobs.append((v, mirrored_peeing_output_path(v, input_root, output_root)))

    console.print(f"[bold]Found {len(jobs)} video(s)[/] under {input_root}")
    if args.dry_run:
        for inp, outp in jobs:
            console.print(f"  {inp}  →  {outp}")
        return

    summary_path = (cwd / args.summary_csv).resolve()
    events_path = (cwd / args.events_csv).resolve()

    batch_wall_start = time.perf_counter()
    bundle = load_peeing_only_models()
    n_ok = 0
    n_fail = 0
    try:
        for i, (inp, outp) in enumerate(jobs):
            console.print(f"[bold cyan][{i + 1}/{len(jobs)}][/] {inp.name}")
            outp.parent.mkdir(parents=True, exist_ok=True)
            rec = run_peeing_pipeline_video(
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
                    f"inference={rec.inference_sec:.2f}s  encode/io={rec.encode_io_sec:.2f}s  "
                    f"peeing_events={len(rec.events)}  → {outp}"
                )
            else:
                n_fail += 1
                console.print(f"  [red]failed[/] {rec.error}")

            append_csv_row(summary_path, SUMMARY_FIELDS, summary_row_dict(rec, models_init_sec=bundle.init_sec))

            if rec.success and rec.events:
                for j, ev in enumerate(rec.events):
                    append_csv_row(
                        events_path,
                        EVENT_FIELDS,
                        {
                            "input_path": str(inp),
                            "output_path": str(outp),
                            "event_index": j,
                            "start_frame": ev.start_frame,
                            "end_frame": ev.end_frame,
                            "start_sec": f"{ev.start_sec:.6f}",
                            "end_sec": f"{ev.end_sec:.6f}",
                            "duration_sec": f"{ev.duration_sec:.6f}",
                            "start_time_str": ev.start_time_str,
                            "end_time_str": ev.end_time_str,
                            "duration_str": ev.duration_str,
                            "n_confirmed_bboxes": ev.n_confirmed_bboxes,
                            "bboxes_json": ev.bboxes_json,
                            "open_interval": ev.open_interval,
                        },
                    )
    finally:
        bundle.cleanup()

    batch_wall_sec = time.perf_counter() - batch_wall_start
    console.print()
    console.print("[bold]Batch done[/]")
    console.print(f"  Videos:  {len(jobs)}  ([green]{n_ok} ok[/], [red]{n_fail} failed[/])")
    console.print(f"  Batch wall (including one model load): {_format_wall_min_sec(batch_wall_sec)}")
    console.print(f"  Summary CSV: [cyan]{summary_path}[/]")
    console.print(f"  Events CSV:  [cyan]{events_path}[/]")


if __name__ == "__main__":
    main()
