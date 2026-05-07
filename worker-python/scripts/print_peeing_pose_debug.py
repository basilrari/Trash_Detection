#!/usr/bin/env python3
"""Print YOLO persons + MediaPipe pose dump for one video frame (tuning / copy-paste).

Uses the **same** ``PeeingDetector`` keyword arguments as ``pipelines.test_pipeline``
(all ``PEEING_*`` values from ``settings.py``).

Run from anywhere; adds ``worker-python`` to ``sys.path``:

  python scripts/print_peeing_pose_debug.py path/to/video.mp4 --frame 1200
  python scripts/print_peeing_pose_debug.py path/to/video.mp4 --frame 0 --no-landmarks

**Sliding window** — one ``update()`` does not fill the alarm window. Options:

  * ``--replay-window`` — seek to frame 0, run ``update()`` for frames ``0 .. --frame``
    (same YOLO cadence as ``--gate-mode``), then print pose JSON for ``--frame`` plus
    ``peeing_sliding_window`` from the last ``update()``.
  * ``--per-frame`` — stream the whole clip (or ``--start`` / ``--end``) and emit **one
    JSON object per line** with ``peeing`` state from each ``update()`` (same settings;
    ``--gate-mode`` defaults to ``yolo``; use ``env`` to follow ``GATE_MODE`` in
    ``settings.py``, or ``off`` for YOLO every frame).
  * ``--instant-hit-range FIRST LAST`` — plain table: one line ``<frame>\\t<yes|no>`` per
    frame (instant pose hit on sampled frames only). Warms up from frame 0, then prints
    FIRST..LAST (e.g. ``600 1000``).

Paste the JSON into a chat if you want thresholds or geometry rules adjusted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

WORKER_ROOT = Path(__file__).resolve().parents[1]
if str(WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKER_ROOT))


def _strip_landmarks(obj: object) -> object:
    if isinstance(obj, dict):
        out: dict[str, object] = {}
        for k, v in obj.items():
            if k == "landmarks_normalized_in_crop" and isinstance(v, list):
                out[k] = f"<{len(v)} points omitted>"
            else:
                out[k] = _strip_landmarks(v)
        return out
    if isinstance(obj, list):
        return [_strip_landmarks(x) for x in obj]
    return obj


def make_peeing_detector_from_settings():
    """Same constructor kwargs as ``pipelines.test_pipeline.run_pipeline``."""
    from models.peeing_detector import PeeingDetector
    from settings import (
        PEEING_ALARM_ENTER_HIT_FRACTION,
        PEEING_ALARM_EXIT_HIT_FRACTION,
        PEEING_ALARM_MIN_SAMPLES,
        PEEING_CROP_MARGIN,
        PEEING_GROIN_DIST_MAX,
        PEEING_GROIN_LOOSE_FACTOR,
        PEEING_MIN_VISIBILITY,
        PEEING_PELVIC_BAND_Y_ABOVE,
        PEEING_PELVIC_BAND_Y_BELOW,
        PEEING_POSE_MATCH_THRESHOLD,
        PEEING_POSE_MODEL_PATH,
        PEEING_POSE_MODEL_URL,
        PEEING_POSE_STRIDE,
        PEEING_SQUAT_DEPTH_SCALE,
        PEEING_SQUAT_HIP_KNEE_GAP_MAX,
        PEEING_STANDING_Y_MARGIN,
        PEEING_WINDOW_SEC,
        PEEING_WRIST_BAND_MIN_VISIBILITY,
    )

    return PeeingDetector(
        pose_stride=PEEING_POSE_STRIDE,
        crop_margin=PEEING_CROP_MARGIN,
        min_visibility=PEEING_MIN_VISIBILITY,
        groin_dist_max=PEEING_GROIN_DIST_MAX,
        groin_loose_factor=PEEING_GROIN_LOOSE_FACTOR,
        wrist_band_min_visibility=PEEING_WRIST_BAND_MIN_VISIBILITY,
        pelvic_band_y_above=PEEING_PELVIC_BAND_Y_ABOVE,
        pelvic_band_y_below=PEEING_PELVIC_BAND_Y_BELOW,
        standing_y_margin=PEEING_STANDING_Y_MARGIN,
        window_sec=PEEING_WINDOW_SEC,
        pose_match_threshold=PEEING_POSE_MATCH_THRESHOLD,
        alarm_enter_hit_fraction=PEEING_ALARM_ENTER_HIT_FRACTION,
        alarm_exit_hit_fraction=PEEING_ALARM_EXIT_HIT_FRACTION,
        alarm_min_samples=PEEING_ALARM_MIN_SAMPLES,
        squat_hip_knee_gap_max=PEEING_SQUAT_HIP_KNEE_GAP_MAX,
        squat_depth_scale=PEEING_SQUAT_DEPTH_SCALE,
        model_path=PEEING_POSE_MODEL_PATH,
        model_url=PEEING_POSE_MODEL_URL,
    )


def peeing_detector_settings_dict() -> dict[str, Any]:
    from settings import (
        PEEING_ALARM_ENTER_HIT_FRACTION,
        PEEING_ALARM_EXIT_HIT_FRACTION,
        PEEING_ALARM_MIN_SAMPLES,
        PEEING_CROP_MARGIN,
        PEEING_GROIN_DIST_MAX,
        PEEING_GROIN_LOOSE_FACTOR,
        PEEING_MIN_VISIBILITY,
        PEEING_PELVIC_BAND_Y_ABOVE,
        PEEING_PELVIC_BAND_Y_BELOW,
        PEEING_POSE_MATCH_THRESHOLD,
        PEEING_POSE_MODEL_PATH,
        PEEING_POSE_STRIDE,
        PEEING_SQUAT_DEPTH_SCALE,
        PEEING_SQUAT_HIP_KNEE_GAP_MAX,
        PEEING_STANDING_Y_MARGIN,
        PEEING_WINDOW_SEC,
        PEEING_WRIST_BAND_MIN_VISIBILITY,
    )

    return {
        "pose_stride": PEEING_POSE_STRIDE,
        "crop_margin": PEEING_CROP_MARGIN,
        "min_visibility": PEEING_MIN_VISIBILITY,
        "groin_dist_max": PEEING_GROIN_DIST_MAX,
        "groin_loose_factor": PEEING_GROIN_LOOSE_FACTOR,
        "wrist_band_min_visibility": PEEING_WRIST_BAND_MIN_VISIBILITY,
        "pelvic_band_y_above": PEEING_PELVIC_BAND_Y_ABOVE,
        "pelvic_band_y_below": PEEING_PELVIC_BAND_Y_BELOW,
        "standing_y_margin": PEEING_STANDING_Y_MARGIN,
        "window_sec": PEEING_WINDOW_SEC,
        "pose_match_threshold": PEEING_POSE_MATCH_THRESHOLD,
        "alarm_enter_hit_fraction": PEEING_ALARM_ENTER_HIT_FRACTION,
        "alarm_exit_hit_fraction": PEEING_ALARM_EXIT_HIT_FRACTION,
        "alarm_min_samples": PEEING_ALARM_MIN_SAMPLES,
        "squat_hip_knee_gap_max": PEEING_SQUAT_HIP_KNEE_GAP_MAX,
        "squat_depth_scale": PEEING_SQUAT_DEPTH_SCALE,
        "pose_model_path": PEEING_POSE_MODEL_PATH,
    }


def peeing_state_summary(p: Any) -> dict[str, Any]:
    from models.peeing_detector import PeeingState

    if not isinstance(p, PeeingState):
        raise TypeError(f"expected PeeingState, got {type(p)}")
    return {
        "active": p.active,
        "status": p.status,
        "score": round(float(p.score), 5),
        "sampled": p.sampled,
        "frame_match": round(float(p.frame_match), 5),
        "edge_enter": p.edge_enter,
        "edge_exit": p.edge_exit,
        "overlay_landmark_paths": len(p.overlay_landmarks),
        "mark_bbox_count": len(p.mark_bboxes),
    }


def _normalize_gate_mode(raw: str) -> Literal["off", "yolo"]:
    s = raw.strip().lower()
    if s in ("off", "yolo"):
        return s  # type: ignore[return-value]
    return "yolo"


def resolve_gate_mode(arg: str) -> Literal["off", "yolo"]:
    if arg == "env":
        from settings import GATE_MODE

        return _normalize_gate_mode(GATE_MODE if GATE_MODE in ("off", "yolo") else "yolo")
    return _normalize_gate_mode(arg)


def run_per_frame_stream(
    *,
    video: Path,
    start: int,
    end: int | None,
    gate_mode: Literal["off", "yolo"],
    progress_every: int,
) -> int:
    import cv2

    from core.types import FrameData
    from core.yolo_stride_gate import YoloStrideGate, YoloStrideGateConfig
    from models.yolo_detector import YoloDetector
    from pipelines.test_pipeline import _filter_scene_detections, _scene_has_activity
    from settings import (
        YOLO_COARSE_STRIDE,
        YOLO_CONFIDENCE,
        YOLO_DENSE_IDLE_MISS_STREAK,
        YOLO_DENSE_STRIDE,
    )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print("Could not open video:", video, file=sys.stderr)
        return 2

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    last_idx = n_frames - 1 if n_frames > 0 else 0
    end_use = end if end is not None else last_idx
    end_use = min(end_use, last_idx)
    if start < 0 or start > end_use:
        print("Invalid --start / --end range.", file=sys.stderr)
        return 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))

    meta = {
        "type": "peeing_per_frame_header",
        "video": str(video.resolve()),
        "fps": fps,
        "frame_count": n_frames,
        "frame_size_wh": [w, h],
        "start": start,
        "end": end_use,
        "gate_mode": gate_mode,
        "yolo_confidence": YOLO_CONFIDENCE,
        "peeing_detector_settings": peeing_detector_settings_dict(),
    }
    print(json.dumps(meta))

    yolo = YoloDetector(conf_threshold=YOLO_CONFIDENCE)
    peeing: PeeingDetector = make_peeing_detector_from_settings()
    gate: YoloStrideGate | None = None
    if gate_mode == "yolo":
        gate = YoloStrideGate(
            YoloStrideGateConfig(
                coarse_stride=YOLO_COARSE_STRIDE,
                dense_stride=YOLO_DENSE_STRIDE,
                dense_idle_miss_streak=YOLO_DENSE_IDLE_MISS_STREAK,
            )
        )

    try:
        frame_idx = start
        while frame_idx <= end_use:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            ts = frame_idx / fps
            run_yolo = True
            scene_dets: list = []
            if gate_mode == "yolo":
                assert gate is not None
                run_yolo = gate.should_run_yolo(frame_idx)
                if run_yolo:
                    fd = FrameData(index=frame_idx, timestamp=ts, image=frame)
                    raw = yolo.detect([fd])[0]
                    scene_dets = _filter_scene_detections(raw)
                    gate.observe(frame_idx, _scene_has_activity(scene_dets, YOLO_CONFIDENCE))
            else:
                fd = FrameData(index=frame_idx, timestamp=ts, image=frame)
                raw = yolo.detect([fd])[0]
                scene_dets = _filter_scene_detections(raw)

            pstate = peeing.update(
                frame,
                scene_dets,
                run_yolo=run_yolo,
                yolo_conf=YOLO_CONFIDENCE,
                timestamp_sec=ts,
            )

            row = {
                "frame_index": frame_idx,
                "timestamp_sec": round(ts, 6),
                "run_yolo": run_yolo,
                "n_scene_detections": len(scene_dets),
                "peeing": peeing_state_summary(pstate),
            }
            print(json.dumps(row))

            if progress_every > 0 and frame_idx % progress_every == 0:
                print(f"... frame {frame_idx}", file=sys.stderr)

            frame_idx += 1
    finally:
        peeing.close()
        cap.release()

    return 0


def run_instant_hit_range(
    *,
    video: Path,
    first: int,
    last: int,
    gate_mode: Literal["off", "yolo"],
) -> int:
    """Warm up from frame 0, then print ``frame<TAB>yes|no`` for instant hit on FIRST..LAST."""
    import cv2
    import numpy as np

    from core.types import FrameData
    from core.yolo_stride_gate import YoloStrideGate, YoloStrideGateConfig
    from models.yolo_detector import YoloDetector
    from pipelines.test_pipeline import _filter_scene_detections, _scene_has_activity
    from settings import (
        YOLO_COARSE_STRIDE,
        YOLO_CONFIDENCE,
        YOLO_DENSE_IDLE_MISS_STREAK,
        YOLO_DENSE_STRIDE,
    )

    if first < 0 or last < first:
        print("--instant-hit-range: need 0 <= FIRST <= LAST", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print("Could not open video:", video, file=sys.stderr)
        return 2

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n_frames > 0 and last >= n_frames:
        print(
            f"--instant-hit-range LAST={last} is past end ({n_frames - 1}).",
            file=sys.stderr,
        )
        return 2

    yolo = YoloDetector(conf_threshold=YOLO_CONFIDENCE)
    peeing = make_peeing_detector_from_settings()
    gate: YoloStrideGate | None = None
    if gate_mode == "yolo":
        gate = YoloStrideGate(
            YoloStrideGateConfig(
                coarse_stride=YOLO_COARSE_STRIDE,
                dense_stride=YOLO_DENSE_STRIDE,
                dense_idle_miss_streak=YOLO_DENSE_IDLE_MISS_STREAK,
            )
        )

    thr = float(peeing.pose_match_threshold)

    def step(frame_idx: int, frame: "np.ndarray") -> Any:
        ts = frame_idx / fps
        run_yolo = True
        scene_dets: list = []
        if gate_mode == "yolo":
            assert gate is not None
            run_yolo = gate.should_run_yolo(frame_idx)
            if run_yolo:
                fd = FrameData(index=frame_idx, timestamp=ts, image=frame)
                raw = yolo.detect([fd])[0]
                scene_dets = _filter_scene_detections(raw)
                gate.observe(frame_idx, _scene_has_activity(scene_dets, YOLO_CONFIDENCE))
        else:
            fd = FrameData(index=frame_idx, timestamp=ts, image=frame)
            raw = yolo.detect([fd])[0]
            scene_dets = _filter_scene_detections(raw)
        return peeing.update(
            frame,
            scene_dets,
            run_yolo=run_yolo,
            yolo_conf=YOLO_CONFIDENCE,
            timestamp_sec=ts,
        )

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
        for i in range(0, first):
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Could not read frame", i, "during warmup", file=sys.stderr)
                return 2
            step(i, frame)

        for i in range(first, last + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Could not read frame", i, file=sys.stderr)
                return 2
            pstate = step(i, frame)
            inst_hit = bool(
                pstate.sampled
                and float(pstate.frame_match) >= thr - 1e-12
            )
            print(f"{i}\t{'yes' if inst_hit else 'no'}")
    finally:
        peeing.close()
        cap.release()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="0-based frame index for snapshot mode (default: 0)",
    )
    parser.add_argument(
        "--no-landmarks",
        action="store_true",
        help="Omit per-landmark x/y/z/visibility arrays (smaller paste)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent for snapshot mode (default: 2; use 0 for one line)",
    )
    parser.add_argument(
        "--per-frame",
        action="store_true",
        help="Emit JSONL: one line per frame with PeeingState (after first header line)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="With --per-frame: first frame index (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="With --per-frame: last frame index inclusive (default: video end)",
    )
    parser.add_argument(
        "--gate-mode",
        choices=("env", "off", "yolo"),
        default="yolo",
        help="YOLO cadence for PeeingDetector.update run_yolo: default yolo=stride gate "
        "(same as worker when GATE_MODE=yolo); env=settings.GATE_MODE; off=every frame",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        metavar="N",
        help="With --per-frame: print progress to stderr every N frames (0=off)",
    )
    parser.add_argument(
        "--replay-window",
        action="store_true",
        help="Snapshot mode only: process frames 0..--frame with same gate as --gate-mode "
        "before printing pose JSON, and include peeing_sliding_window from last update()",
    )
    parser.add_argument(
        "--instant-hit-range",
        nargs=2,
        type=int,
        metavar=("FIRST", "LAST"),
        help="Print one line per frame FIRST..LAST: '<frame>\\t<yes|no>' (yes = pose sampled "
        "and best instant score >= pose_match_threshold). Warms up from frame 0 first.",
    )
    args = parser.parse_args()

    if args.per_frame and args.replay_window:
        print("Use only one of --per-frame or --replay-window.", file=sys.stderr)
        return 2
    if args.instant_hit_range is not None and (
        args.per_frame or args.replay_window
    ):
        print(
            "--instant-hit-range cannot be combined with --per-frame or --replay-window.",
            file=sys.stderr,
        )
        return 2

    gate_mode = resolve_gate_mode(args.gate_mode)

    if args.per_frame:
        return run_per_frame_stream(
            video=args.video,
            start=args.start,
            end=args.end,
            gate_mode=gate_mode,
            progress_every=max(0, args.progress_every),
        )

    if args.instant_hit_range is not None:
        lo, hi = args.instant_hit_range
        return run_instant_hit_range(
            video=args.video,
            first=lo,
            last=hi,
            gate_mode=gate_mode,
        )

    import cv2

    from core.types import FrameData
    from core.yolo_stride_gate import YoloStrideGate, YoloStrideGateConfig
    from models.yolo_detector import YoloDetector
    from pipelines.test_pipeline import _filter_scene_detections, _scene_has_activity
    from settings import (
        YOLO_COARSE_STRIDE,
        YOLO_CONFIDENCE,
        YOLO_DENSE_IDLE_MISS_STREAK,
        YOLO_DENSE_STRIDE,
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print("Could not open video:", args.video, file=sys.stderr)
        return 2

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    peeing: PeeingDetector = make_peeing_detector_from_settings()

    try:
        if args.replay_window:
            if args.frame < 0:
                print("--frame must be >= 0", file=sys.stderr)
                return 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
            yolo = YoloDetector(conf_threshold=YOLO_CONFIDENCE)
            gate: YoloStrideGate | None = None
            if gate_mode == "yolo":
                gate = YoloStrideGate(
                    YoloStrideGateConfig(
                        coarse_stride=YOLO_COARSE_STRIDE,
                        dense_stride=YOLO_DENSE_STRIDE,
                        dense_idle_miss_streak=YOLO_DENSE_IDLE_MISS_STREAK,
                    )
                )
            peeing.reset()
            last_frame = None
            last_yolo_raw: list = []
            last_pstate = None
            for i in range(0, args.frame + 1):
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("Could not read frame", i, "during replay", file=sys.stderr)
                    return 2
                ts = i / fps
                run_yolo = True
                scene_dets: list = []
                if gate_mode == "yolo":
                    assert gate is not None
                    run_yolo = gate.should_run_yolo(i)
                    if run_yolo:
                        fd = FrameData(index=i, timestamp=ts, image=frame)
                        raw = yolo.detect([fd])[0]
                        last_yolo_raw = raw
                        scene_dets = _filter_scene_detections(raw)
                        gate.observe(i, _scene_has_activity(scene_dets, YOLO_CONFIDENCE))
                else:
                    fd = FrameData(index=i, timestamp=ts, image=frame)
                    raw = yolo.detect([fd])[0]
                    last_yolo_raw = raw
                    scene_dets = _filter_scene_detections(raw)
                last_pstate = peeing.update(
                    frame,
                    scene_dets,
                    run_yolo=run_yolo,
                    yolo_conf=YOLO_CONFIDENCE,
                    timestamp_sec=ts,
                )
                last_frame = frame
            assert last_frame is not None
            frame = last_frame
            dets = last_yolo_raw
            pstate_snap = last_pstate
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(args.frame))
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Could not read frame", args.frame, file=sys.stderr)
                return 2
            yolo = YoloDetector(conf_threshold=YOLO_CONFIDENCE)
            dets = yolo.detect(
                [FrameData(index=args.frame, timestamp=args.frame / fps, image=frame)]
            )[0]
            scene = _filter_scene_detections(dets)
            pstate_snap = peeing.update(
                frame,
                scene,
                run_yolo=True,
                yolo_conf=YOLO_CONFIDENCE,
                timestamp_sec=args.frame / fps,
            )

        reports = peeing.debug_person_reports(frame, dets, yolo_conf=YOLO_CONFIDENCE)
    finally:
        peeing.close()
    cap.release()

    h, w = frame.shape[:2]
    payload: dict[str, object] = {
        "video": str(args.video.resolve()),
        "frame_index": args.frame,
        "frame_size_wh": [w, h],
        "yolo_confidence_setting": YOLO_CONFIDENCE,
        "gate_mode": gate_mode,
        "replay_window": bool(args.replay_window),
        "peeing_detector_settings": peeing_detector_settings_dict(),
        "peeing_sliding_window": peeing_state_summary(pstate_snap),
        "person_pose_reports": reports,
    }

    if args.no_landmarks:
        payload = _strip_landmarks(payload)  # type: ignore[assignment]

    indent = None if args.indent == 0 else args.indent
    print(json.dumps(payload, indent=indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
