#!/usr/bin/env bash
# Run pipeline benchmark (from worker-python/). All YOLO/LP/RF-DETR paths come from settings.py.
set -euo pipefail
VIDEO="${1:-inputs/Peeing 96s.mp4}"
OUTP="${2:-outputs}"
mkdir -p "$OUTP"
python worker.py "$VIDEO" -o "$OUTP/bench_run.mp4"
