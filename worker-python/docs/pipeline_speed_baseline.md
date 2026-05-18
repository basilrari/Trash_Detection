# Pipeline speed baseline

Use this sheet when comparing runs **before and after** changing `settings.py` or pipeline code.

## How to record a baseline

1. Pick one **fixed** input (single file or the same batch manifest) and keep `OUTPUT_VIDEO` distinct per run if you need to keep artifacts.
2. Run the pipeline from `worker-python/` the same way each time (same machine load if possible).
3. Copy the printed **Step timings (cumulative)** block into the table below.
4. Note the **git commit** hash and any local edits not committed.

## Baseline table (example row from a prior run)

| Run id | Date | Commit | Input | Wall s | Init s | YOLO s | RF-DETR s | Peeing s | LP s | OCR s | LP crops | LP batches | avg crops/batch | emit_barriers | latency_flushes | queue_full_flushes | eof_rounds | OCR calls | locked OCR skips | prefilter skips |
|--------|------|--------|-------|--------|--------|--------|-----------|----------|------|-------|----------|------------|-----------------|---------------|-----------------|-------------------|------------|-----------|------------------|-----------------|
| example | — | — | 550 sampled frames | 109.63 | 16.15 | 20.36 | 11.25 | 0.04 | 16.95 | 16.00 | 1399 | 376 | 3.72 | 323 | 0 | — | — | — | — | — |

Fill **emit_barriers**, **latency_flushes**, **queue_full_flushes**, **eof_rounds** from the
``[dim]LP cross-frame:`` line, and **OCR calls** / **locked_reuse_skips** / **prefilter_skipped** from the
``[dim]OCR detail:`` line in the timing summary.

## Rows for your A/B tests

Add one row per experiment (change **one** knob at a time when isolating regressions).

| Run id | Change | Wall s | LP s | OCR s | Notes |
|--------|--------|--------|------|-------|-------|
| A1 | (baseline) | | | | |
| A2 | … | | | | |
