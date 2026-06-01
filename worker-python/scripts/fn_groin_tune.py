#!/usr/bin/env python3
"""
Sweep ``PEEING_HAND_GROIN_Y_THRESHOLD`` on evaluation clips and write a TP/FN CSV.

FN1–FN3 are **false negatives** in production: ground truth is **peeing = yes**.
For each (video, threshold) run, a **TP** means peeing was confirmed; **FN** means it was missed.

From ``worker-python/``::

  # Default: manifest ``inputs/fn_eval_manifest.csv`` (FN* + TP* + FP*) → outputs/fn_groin_eval.csv
  python scripts/fn_groin_tune.py

  # Annotated pose MP4s + eval CSV (TP/FN/FP/TN per clip) → outputs/fn_tune/
  python scripts/fn_groin_tune.py --viz --threshold 1

  # Custom eval CSV path
  python scripts/fn_groin_tune.py --viz --threshold 1 --csv outputs/my_eval.csv

  # Groin threshold × confirm-time grid (14 clips × 10 × 6 = 840 runs)
  python scripts/fn_groin_tune.py \\
    --thresholds 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 \\
    --seconds-required 5 6 7 8 9 10 \\
    --csv outputs/fn_groin_eval_thr_sec.csv

  # Confirm-time only (5–10 s), default groin from settings.py
  python scripts/fn_groin_tune.py --seconds-required 5 6 7 8 9 10 \\
    --csv outputs/fn_groin_eval_sec.csv

  # Viz only, no CSV
  python scripts/fn_groin_tune.py --viz --no-csv

After changing one-side standing/groin in ``peeing_detector.py``, re-run sweep + viz on FN clips
to confirm TP in ``outputs/fn_groin_eval.csv`` and ``standing (sides=L|R)`` on overlay.
  python scripts/fn_groin_tune.py --viz --threshold 0.12

  # Custom grid or manifest
  python scripts/fn_groin_tune.py --csv outputs/my_sweep.csv
  python scripts/fn_groin_tune.py --manifest inputs/fn_eval_manifest.csv

  # Pose overlay for one threshold (optional)
  python scripts/fn_groin_tune.py --viz --threshold 0.12
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_WORKER = Path(__file__).resolve().parent.parent
if str(_WORKER) not in sys.path:
    sys.path.insert(0, str(_WORKER))

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from pipelines.peeing_pipeline import (
    PeeingPipelineOptions,
    load_peeing_only_models,
    run_peeing_pipeline_video,
)
from settings import (
    OUTPUTS_DIR,
    PEEING_HAND_GROIN_Y_THRESHOLD,
    PEEING_SECONDS_REQUIRED,
)

console = Console()

DEFAULT_MANIFEST = _WORKER / "inputs" / "fn_eval_manifest.csv"
EVAL_GLOB_PATTERNS = (
    "FN*.mp4",
    "FP*.mp4",
    "TP*.mp4",
    "FN*.avi",
    "FP*.avi",
    "TP*.avi",
    "FN*.mov",
    "FP*.mov",
    "TP*.mov",
)

# Full grid: 0.05 … 0.20 step 0.01 (every value in range).
DEFAULT_THRESHOLDS: list[float] = [round(0.05 + i * 0.01, 2) for i in range(16)]

CSV_FIELDS = [
    "video",
    "clip_tag",
    "video_path",
    "hand_groin_y_threshold",
    "seconds_required",
    "should_detect_peeing",
    "ground_truth_peeing",
    "predicted_peeing",
    "result",
    "is_correct",
    "is_tp",
    "peeing_event_count",
    "wall_sec",
    "notes",
]


def _parse_bool(s: str) -> bool:
    v = str(s).strip().lower()
    if v in ("1", "true", "yes", "y", "peeing", "positive", "pos"):
        return True
    if v in ("0", "false", "no", "n", "negative", "neg", "none"):
        return False
    raise ValueError(f"expected yes/no, got {s!r}")


def _load_manifest(path: Path) -> dict[str, bool]:
    """Map video file name → ground_truth_peeing (should confirm peeing)."""
    out: dict[str, bool] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"empty manifest: {path}")
        cols = {c.strip().lower(): c for c in reader.fieldnames if c}
        vid_col = cols.get("video") or cols.get("video_path") or cols.get("file")
        gt_col = (
            cols.get("ground_truth_peeing")
            or cols.get("ground_truth")
            or cols.get("label")
            or cols.get("expected_peeing")
        )
        if not vid_col or not gt_col:
            raise ValueError(
                f"manifest needs video + ground_truth_peeing columns: {path}"
            )
        for row in reader:
            raw = (row.get(vid_col) or "").strip()
            if not raw:
                continue
            name = Path(raw).name
            out[name] = _parse_bool(row.get(gt_col) or "")
    return out


def _discover_eval_videos_in_inputs() -> list[Path]:
    """All ``FN*`` / ``TP*`` / ``FP*`` clips directly under ``inputs/`` (sorted by name)."""
    inputs_dir = _WORKER / "inputs"
    found: list[Path] = []
    for pat in EVAL_GLOB_PATTERNS:
        found.extend(inputs_dir.glob(pat))
    return sorted({p.resolve() for p in found}, key=lambda p: p.name)


def _resolve_manifest_path(manifest: Path | None) -> Path | None:
    if manifest is not None:
        p = manifest.expanduser().resolve()
        return p if p.is_file() else None
    return DEFAULT_MANIFEST if DEFAULT_MANIFEST.is_file() else None


def _resolve_videos(
    *,
    video: str | None,
    videos: list[Path] | None,
    manifest: Path | None,
) -> tuple[list[Path], dict[str, bool] | None]:
    manifest_path = _resolve_manifest_path(manifest)
    manifest_map = (
        _load_manifest(manifest_path) if manifest_path is not None else None
    )

    if video:
        p = Path(video).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"Video not found: {p}")
        return [p], manifest_map

    if videos:
        paths = [Path(v).expanduser().resolve() for v in videos]
    elif manifest_map is not None:
        paths = []
        for name in sorted(manifest_map.keys()):
            cand = _WORKER / "inputs" / name
            if not cand.is_file():
                cand = Path(name).expanduser().resolve()
            if not cand.is_file():
                raise SystemExit(f"Manifest video not found: {name}")
            paths.append(cand)
    else:
        paths = _discover_eval_videos_in_inputs()
        if not paths:
            raise SystemExit(
                "No eval videos found. Add FN*.mp4 / TP*.mp4 / FP*.mp4 under inputs/ "
                f"or create {DEFAULT_MANIFEST}"
            )

    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit("Video not found:\n  " + "\n  ".join(str(p) for p in missing))
    return paths, manifest_map


def _ground_truth_for(
    vpath: Path, manifest_map: dict[str, bool] | None
) -> bool:
    if manifest_map is not None and vpath.name in manifest_map:
        return manifest_map[vpath.name]
    stem = vpath.stem.upper()
    if stem.startswith("FN") or stem.startswith("TP"):
        return True
    if stem.startswith("FP"):
        return False
    raise SystemExit(
        f"No ground truth for {vpath.name}. Add a row to {DEFAULT_MANIFEST} "
        "(ground_truth_peeing: yes = should detect peeing, no = should not)."
    )


def _classify(ground_truth: bool, predicted: bool) -> tuple[str, bool]:
    """Confusion label and whether the prediction matches ground truth (TP or TN)."""
    if ground_truth and predicted:
        return "TP", True
    if ground_truth and not predicted:
        return "FN", False
    if not ground_truth and predicted:
        return "FP", False
    return "TN", True


def _thr_equal(a: float, b: float, *, eps: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= eps


def _row_is_correct(row: dict[str, str | float | int | bool]) -> bool:
    """True for TP or TN — use ground truth + prediction, not ``result == TP``."""
    gt_raw = row.get("should_detect_peeing", row.get("ground_truth_peeing", ""))
    pr_raw = row.get("predicted_peeing", "")
    gt = str(gt_raw).strip().lower() in ("1", "true", "yes", "y")
    pred = str(pr_raw).strip().lower() in ("1", "true", "yes", "y")
    return _classify(gt, pred)[1]


def _find_eval_row(
    rows: list[dict[str, str | float | int | bool]],
    *,
    video_name: str,
    thr: float,
    seconds_required: int,
) -> dict[str, str | float | int | bool] | None:
    for r in rows:
        if r["video"] != video_name:
            continue
        if not _thr_equal(r["hand_groin_y_threshold"], thr):
            continue
        if int(r["seconds_required"]) != int(seconds_required):
            continue
        return r
    return None


def _resolve_thresholds(args: argparse.Namespace) -> list[float]:
    if args.thresholds:
        return [float(t) for t in args.thresholds]
    if args.threshold is not None:
        return [float(args.threshold)]
    if args.seconds_required and not args.thresholds:
        return [float(PEEING_HAND_GROIN_Y_THRESHOLD)]
    return [float(t) for t in DEFAULT_THRESHOLDS]


def _resolve_seconds_list(args: argparse.Namespace) -> list[int]:
    if args.seconds_required:
        return [max(1, int(s)) for s in args.seconds_required]
    return [max(1, int(PEEING_SECONDS_REQUIRED))]


def _count_correct_rows(
    rows: list[dict[str, str | float | int | bool]],
) -> tuple[int, int, int]:
    """Returns (total_correct, n_tp, n_tn)."""
    n_tp = n_tn = 0
    for r in rows:
        if not _row_is_correct(r):
            continue
        if str(r["result"]) == "TP":
            n_tp += 1
        elif str(r["result"]) == "TN":
            n_tn += 1
    return n_tp + n_tn, n_tp, n_tn


def _clip_tag_from_name(video_name: str) -> str:
    """Filename prefix for eval sets: ``FN1.mp4`` → ``FN``, etc."""
    stem = Path(video_name).stem.upper()
    for prefix in ("FN", "FP", "TP"):
        if stem.startswith(prefix):
            return prefix
    return ""


def _eval_notes(result: str, clip_tag: str, ground_truth: bool) -> str:
    """Short plain-English line for the CSV ``notes`` column."""
    should = "should detect peeing" if ground_truth else "should NOT detect peeing"
    if result == "TP":
        if clip_tag == "FN":
            return f"Prod miss ({should}); model detected — fixed"
        if clip_tag == "TP":
            return f"Prod hit ({should}); model detected — OK"
        return f"{should}; model detected — OK"
    if result == "FN":
        if clip_tag in ("FN", "TP"):
            return f"Prod miss/hit ({should}); model missed — still wrong"
        return f"{should}; model missed"
    if result == "FP":
        return f"{should}; model fired — false alarm"
    return f"{should}; model silent — OK"


def _make_eval_row(
    *,
    vpath: Path,
    thr: float,
    seconds_required: int,
    ground_truth: bool,
    predicted: bool,
    event_count: int,
    wall_sec: float,
) -> dict[str, str | float | int | bool]:
    result, prediction_ok = _classify(ground_truth, predicted)
    gt_s = "yes" if ground_truth else "no"
    tag = _clip_tag_from_name(vpath.name)
    return {
        "video": vpath.name,
        "clip_tag": tag,
        "video_path": str(vpath),
        "hand_groin_y_threshold": thr,
        "seconds_required": int(seconds_required),
        "should_detect_peeing": gt_s,
        "ground_truth_peeing": gt_s,
        "predicted_peeing": "yes" if predicted else "no",
        "result": result,
        "is_correct": "yes" if prediction_ok else "no",
        "is_tp": "yes" if result == "TP" else "no",
        "peeing_event_count": event_count,
        "wall_sec": round(wall_sec, 2),
        "notes": _eval_notes(result, tag, ground_truth),
    }


def _write_eval_csv(rows: list[dict[str, str | float | int | bool]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _print_sweep_correct_matrix(
    rows: list[dict[str, str | float | int | bool]],
    *,
    videos: list[Path],
    thresholds: list[float],
    seconds_list: list[int],
) -> None:
    """Compact correct-count grid when sweeping both threshold and confirm seconds."""
    if len(thresholds) <= 1 and len(seconds_list) <= 1:
        return
    n = len(videos)
    table = Table(
        title="Correct clips (TP+TN) per hand_groin threshold × PEEING_SECONDS_REQUIRED"
    )
    table.add_column("thr \\ sec", justify="right")
    for sec in seconds_list:
        table.add_column(f"{sec}s", justify="center")
    for thr in thresholds:
        cells: list[str] = [f"{thr:.2f}"]
        for sec in seconds_list:
            sub = [
                r
                for r in rows
                if _thr_equal(r["hand_groin_y_threshold"], thr)
                and int(r["seconds_required"]) == int(sec)
            ]
            ok_n, n_tp, n_tn = _count_correct_rows(sub)
            cells.append(f"{ok_n}/{n} ({n_tp}+{n_tn})")
        table.add_row(*cells)
    console.print(table)


def _print_threshold_video_table(
    rows: list[dict[str, str | float | int | bool]],
    *,
    videos: list[Path],
    thresholds: list[float],
    seconds_list: list[int],
) -> None:
    """Per-video results for one ``seconds_required`` (or single-value sweep)."""
    sec = seconds_list[0]
    table = Table(
        title=(
            f"Result per video at each threshold "
            f"(seconds_required={sec}) — correct = TP or TN"
        )
    )
    table.add_column("threshold", justify="right")
    for vpath in videos:
        table.add_column(vpath.stem, justify="center")
    table.add_column("correct", justify="right")
    for thr in thresholds:
        cells = [f"{thr:.2f}"]
        thr_rows = [
            r
            for r in rows
            if _thr_equal(r["hand_groin_y_threshold"], thr)
            and int(r["seconds_required"]) == int(sec)
        ]
        ok_n, n_tp, n_tn = _count_correct_rows(thr_rows)
        for vpath in videos:
            row = _find_eval_row(
                rows, video_name=vpath.name, thr=thr, seconds_required=sec
            )
            if row is None:
                cells.append("?")
                continue
            cells.append(str(row["result"]))
        cells.append(f"{ok_n}/{len(videos)} ({n_tp} TP · {n_tn} TN)")
        table.add_row(*cells)
    console.print(table)


def _print_seconds_video_table(
    rows: list[dict[str, str | float | int | bool]],
    *,
    videos: list[Path],
    thresholds: list[float],
    seconds_list: list[int],
) -> None:
    """Per-video results for one threshold across confirm-second values."""
    thr = thresholds[0]
    table = Table(
        title=(
            f"Result per video at each seconds_required "
            f"(hand_groin={thr:.2f}) — correct = TP or TN"
        )
    )
    table.add_column("sec", justify="right")
    for vpath in videos:
        table.add_column(vpath.stem, justify="center")
    table.add_column("correct", justify="right")
    for sec in seconds_list:
        cells = [str(sec)]
        sec_rows = [
            r
            for r in rows
            if _thr_equal(r["hand_groin_y_threshold"], thr)
            and int(r["seconds_required"]) == int(sec)
        ]
        ok_n, n_tp, n_tn = _count_correct_rows(sec_rows)
        for vpath in videos:
            row = _find_eval_row(
                rows, video_name=vpath.name, thr=thr, seconds_required=sec
            )
            if row is None:
                cells.append("?")
                continue
            cells.append(str(row["result"]))
        cells.append(f"{ok_n}/{len(videos)} ({n_tp} TP · {n_tn} TN)")
        table.add_row(*cells)
    console.print(table)


def _print_eval_summary(
    rows: list[dict[str, str | float | int | bool]],
    *,
    thr: float | None,
    seconds_required: int | None,
    out_csv: Path,
) -> None:
    """Terminal table: per-clip TP/FN/FP/TN and counts."""
    parts: list[str] = []
    if thr is not None:
        parts.append(f"threshold {thr:.2f}")
    if seconds_required is not None:
        parts.append(f"seconds_required {seconds_required}")
    thr_s = f" @ {', '.join(parts)}" if parts else ""
    table = Table(title=f"Eval summary{thr_s}")
    table.add_column("video")
    table.add_column("tag", justify="center")
    table.add_column("should", justify="center")
    table.add_column("predicted", justify="center")
    table.add_column("result", justify="center")
    table.add_column("OK?", justify="center")
    table.add_column("events", justify="right")

    counts: dict[str, int] = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    ok_n, n_tp, n_tn = _count_correct_rows(rows)
    for row in rows:
        res = str(row["result"])
        counts[res] = counts.get(res, 0) + 1
        ok = _row_is_correct(row)
        table.add_row(
            str(row["video"]),
            str(row.get("clip_tag") or "—"),
            str(row.get("should_detect_peeing") or row.get("ground_truth_peeing")),
            str(row["predicted_peeing"]),
            res,
            "[green]yes[/]" if ok else "[red]no[/]",
            str(row["peeing_event_count"]),
        )
    table.add_row(
        "—", "—", "—", "—", "—", f"{ok_n}/{len(rows)} ({n_tp} TP · {n_tn} TN)", "—"
    )
    console.print(table)
    console.print(
        "[dim]Counts:[/] "
        f"TP={counts.get('TP', 0)}  FN={counts.get('FN', 0)}  "
        f"FP={counts.get('FP', 0)}  TN={counts.get('TN', 0)}"
    )
    by_tag: dict[str, list[str]] = {}
    for row in rows:
        tag = str(row.get("clip_tag") or "?")
        by_tag.setdefault(tag, []).append(
            f"{row['video']}→{row['result']}"
        )
    for tag in ("FN", "TP", "FP"):
        if tag in by_tag:
            console.print(f"[dim]{tag} clips:[/] {', '.join(by_tag[tag])}")
    console.print(f"[green]CSV ({len(rows)} rows):[/] {out_csv}")


def run_sweep(args: argparse.Namespace) -> None:
    manifest_path = _resolve_manifest_path(args.manifest)
    videos, manifest_map = _resolve_videos(
        video=args.video,
        videos=args.videos,
        manifest=args.manifest,
    )
    thresholds = _resolve_thresholds(args)
    seconds_list = _resolve_seconds_list(args)
    out_csv = Path(
        args.csv or (_WORKER / OUTPUTS_DIR / "fn_groin_eval.csv")
    ).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    total_runs = len(videos) * len(thresholds) * len(seconds_list)
    src = str(manifest_path) if manifest_path else "FN*/TP*/FP* glob in inputs/"
    console.print(
        f"[bold]Peeing eval sweep[/]  runs={total_runs}  "
        f"({len(videos)} videos × {len(thresholds)} thresholds × "
        f"{len(seconds_list)} confirm seconds)"
    )
    console.print(f"[dim]Video list from:[/] {src}")
    console.print(
        f"[dim]Hand→groin thresholds:[/] {thresholds[0]:.2f} … {thresholds[-1]:.2f} "
        f"({len(thresholds)} values)"
    )
    console.print(
        f"[dim]PEEING_SECONDS_REQUIRED:[/] {seconds_list[0]} … {seconds_list[-1]} "
        f"({len(seconds_list)} values)"
    )
    console.print(
        "[yellow]Sweep mode does not write annotated MP4 files[/] (inference only, faster)."
    )
    console.print(f"[dim]Results CSV:[/] {out_csv}")
    console.print(
        "[dim]Pose videos:[/] use "
        "`python scripts/fn_groin_tune.py --viz --threshold 0.12` "
        f"→ {_WORKER / OUTPUTS_DIR / 'fn_tune'}/"
    )
    console.print()

    rows: list[dict[str, str | float | int | bool]] = []
    bundle = load_peeing_only_models(
        options=PeeingPipelineOptions(
            write_output=False,
            collect_pose_viz=False,
            quiet=True,
            show_frame_progress=True,
        ),
    )
    run_i = 0
    try:
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Starting sweep…", total=total_runs)
            for vpath in videos:
                gt = _ground_truth_for(vpath, manifest_map)
                for thr in thresholds:
                    for sec in seconds_list:
                        run_i += 1
                        pct = 100.0 * (run_i - 1) / max(total_runs, 1)
                        label = (
                            f"Run {run_i}/{total_runs} ({pct:.0f}%) — "
                            f"{vpath.name} — thr={thr:.2f} sec={sec} — inferring…"
                        )
                        progress.update(task, description=label, completed=run_i - 1)
                        bundle.peeing.hand_groin_y_threshold = float(thr)
                        bundle.peeing.seconds_required = int(sec)
                        opts = PeeingPipelineOptions(
                            hand_groin_y_threshold=thr,
                            seconds_required=sec,
                            write_output=False,
                            collect_pose_viz=False,
                            draw_pose=False,
                            quiet=True,
                            show_frame_progress=True,
                            progress_label=f"{vpath.stem} thr={thr:.2f} sec={sec}",
                        )
                        bundle.peeing.reset()
                        bundle.yolo.reset_inference_batch_stats()
                        bundle.peeing.reset_inference_batch_stats()
                        rec = run_peeing_pipeline_video(
                            bundle,
                            str(vpath),
                            str(out_csv),
                            per_video_times_init_sec=0.0,
                            models_init_sec=bundle.init_sec if run_i == 1 else 0.0,
                            abort_on_error=True,
                            pipeline_options=opts,
                        )
                        predicted = len(rec.events) > 0
                        row = _make_eval_row(
                            vpath=vpath,
                            thr=thr,
                            seconds_required=sec,
                            ground_truth=gt,
                            predicted=predicted,
                            event_count=len(rec.events),
                            wall_sec=rec.wall_sec,
                        )
                        rows.append(row)
                        tag = str(row["result"])
                        ok = _row_is_correct(row)
                        progress.console.print(
                            f"  [dim]done[/] {vpath.name} thr={thr:.2f} sec={sec} → "
                            f"{'[green]' + tag + '[/]' if ok else '[red]' + tag + '[/]'} "
                            f"(events={len(rec.events)}, {rec.wall_sec:.1f}s)"
                        )
                        progress.update(task, advance=1)
            progress.update(
                task,
                description=f"Sweep complete ({total_runs}/{total_runs})",
                completed=total_runs,
            )
    finally:
        bundle.cleanup()

    for r in rows:
        r["is_correct"] = "yes" if _row_is_correct(r) else "no"

    _print_sweep_correct_matrix(
        rows, videos=videos, thresholds=thresholds, seconds_list=seconds_list
    )
    if len(seconds_list) == 1:
        _print_threshold_video_table(
            rows, videos=videos, thresholds=thresholds, seconds_list=seconds_list
        )
    elif len(thresholds) == 1:
        _print_seconds_video_table(
            rows, videos=videos, thresholds=thresholds, seconds_list=seconds_list
        )
    _write_eval_csv(rows, out_csv)
    console.print(f"[green]Sweep CSV ({len(rows)} rows):[/] {out_csv}")


def run_viz(args: argparse.Namespace) -> None:
    thr = (
        float(args.threshold)
        if args.threshold is not None
        else float(PEEING_HAND_GROIN_Y_THRESHOLD)
    )
    sec_list = _resolve_seconds_list(args)
    if len(sec_list) != 1:
        raise SystemExit(
            "--viz supports one --seconds-required value only "
            "(omit it to use settings.py, or pass e.g. --seconds-required 5)."
        )
    sec = sec_list[0]
    videos, manifest_map = _resolve_videos(
        video=args.video,
        videos=args.videos,
        manifest=args.manifest,
    )
    write_csv = not bool(args.no_csv)
    out_dir = Path(args.output_dir or (_WORKER / OUTPUTS_DIR / "fn_tune")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv: Path | None = None
    if write_csv:
        out_csv = Path(
            args.csv or (out_dir / f"eval_summary_thr{thr:.2f}_sec{sec}.csv")
        ).resolve()

    opts = PeeingPipelineOptions(
        hand_groin_y_threshold=thr,
        seconds_required=sec,
        collect_pose_viz=True,
        draw_pose=True,
        write_output=True,
        quiet=False,
        show_frame_progress=True,
    )
    console.print(
        f"[bold]Pose viz mode[/]  threshold={thr:.2f}  seconds_required={sec}  "
        f"videos={len(videos)}  output dir: {out_dir}"
    )
    if write_csv and out_csv is not None:
        console.print(f"[dim]Results CSV:[/] {out_csv}")

    rows: list[dict[str, str | float | int | bool]] = []
    bundle = load_peeing_only_models(options=opts)
    try:
        for n, vpath in enumerate(videos, start=1):
            out_path = out_dir / f"{vpath.stem}_thr{thr:.2f}_sec{sec}_pose{vpath.suffix}"
            bundle.peeing.hand_groin_y_threshold = float(thr)
            bundle.peeing.seconds_required = int(sec)
            bundle.peeing.reset()
            opts.progress_label = f"{vpath.stem} thr={thr:.2f} ({n}/{len(videos)})"
            console.print(
                f"[cyan]Viz {n}/{len(videos)}[/]  {vpath.name}  →  {out_path.resolve()}"
            )
            rec = run_peeing_pipeline_video(
                bundle,
                str(vpath),
                str(out_path),
                per_video_times_init_sec=0.0,
                models_init_sec=bundle.init_sec if n == 1 else 0.0,
                abort_on_error=True,
                pipeline_options=opts,
            )
            if write_csv:
                gt = _ground_truth_for(vpath, manifest_map)
                predicted = len(rec.events) > 0
                row = _make_eval_row(
                    vpath=vpath,
                    thr=thr,
                    seconds_required=sec,
                    ground_truth=gt,
                    predicted=predicted,
                    event_count=len(rec.events),
                    wall_sec=rec.wall_sec,
                )
                rows.append(row)
                tag = str(row["result"])
                ok = _row_is_correct(row)
                console.print(
                    f"  [dim]eval[/] {vpath.name} → "
                    f"{'[green]' + tag + '[/]' if ok else '[red]' + tag + '[/]'} "
                    f"(events={len(rec.events)}, {rec.wall_sec:.1f}s)"
                )
    finally:
        bundle.cleanup()
    console.print(f"[green]Pose videos saved under[/] {out_dir.resolve()}")

    if write_csv and out_csv is not None and rows:
        for r in rows:
            r["is_correct"] = "yes" if _row_is_correct(r) else "no"
        _write_eval_csv(rows, out_csv)
        _print_eval_summary(
            rows, thr=thr, seconds_required=sec, out_csv=out_csv
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep hand-to-groin threshold and/or PEEING_SECONDS_REQUIRED; "
            "CSV with TP/FN/FP/TN per video."
        ),
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Write pose-annotated MP4s for --threshold (default mode is full CSV sweep).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="With --viz: skip eval CSV and end summary (default: write eval_summary_thr*.csv).",
    )
    parser.add_argument(
        "--with-csv",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help=(
            "Hand-to-groin Y threshold for --viz "
            f"(default: settings.py = {PEEING_HAND_GROIN_Y_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        help=(
            "Sweep grid (default: 0.05 … 0.20 when neither this nor --seconds-required). "
            "Example: --thresholds 0.01 0.02 … 0.10"
        ),
    )
    parser.add_argument(
        "--seconds-required",
        nargs="+",
        type=int,
        metavar="SEC",
        help=(
            "Sweep PEEING_SECONDS_REQUIRED (consecutive good seconds before confirm). "
            f"Example: --seconds-required 5 6 7 8 9 10 (default: {PEEING_SECONDS_REQUIRED})."
        ),
    )
    parser.add_argument("--video", help="Single video path.")
    parser.add_argument("--videos", nargs="+", type=Path, help="Explicit video paths.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            f"CSV: video, ground_truth_peeing (yes/no). "
            f"Default: {DEFAULT_MANIFEST} if present, else FN*/TP*/FP* in inputs/."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help=(
            "Eval CSV path (viz default: outputs/fn_tune/eval_summary_thr<THR>.csv; "
            "sweep default: outputs/fn_groin_eval.csv)."
        ),
    )
    parser.add_argument("--output-dir", help="Output dir for --viz MP4s.")

    args = parser.parse_args()
    if args.viz:
        run_viz(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
