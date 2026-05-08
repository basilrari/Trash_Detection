# YOLO micro-batching (gated pipeline) — plan

Saved for later implementation. **Not implemented** as of this document.

## Goal

Batch **scene YOLO** in the `GATE_MODE=yolo` path when it is **safe and correct**, without changing gate semantics unless a separate **speculative / lookahead** mode is explicitly added later.

## Why batching is limited (short)

- `should_run_yolo(i)` can depend on `observe()` from **earlier** YOLO runs (dense on/off, idle miss streak).
- So you **cannot** skip ahead and batch future gated indices **without** intermediate observes, unless rules change.
- **Dense “pattern”** (e.g. `n, n+2, n+4, …`) is only valid **while dense stays on**; each result can end dense, so the future list is not fixed until you process in order.

## When large batches are realistic

| Situation | Typical batch behavior |
|-----------|-------------------------|
| `YOLO_COARSE_STRIDE == 1` | Every frame is a YOLO frame → can accumulate up to **B** consecutive reads (subject to RAM / optional cap). |
| Sparse coarse (e.g. 8–15) + dense | Often **must commit** after a small pending set because a **non-YOLO** frame or **gate state** needs prior `observe` → expect **batch size 1** frequently. |

## Behavioral spec (before coding)

Define **`YOLO_MICRO_BATCH`** (target max batch, e.g. 8) and **commit** rules in priority order:

1. **EOF** — Run `yolo.detect` on all remaining pending (variable size). No padding with fake **video** frames for YOLO (Ultralytics accepts variable list length).
2. **Must-unblock gate** — If computing `should_run_yolo` for the next frame requires prior `observe` from pending YOLO frames, **commit** pending first (often size 1 when coarse > 1).
3. **Wait-for-fill** — If safe to wait and `len(pending) < B`, keep accumulating until `len == B`, EOF, or must-unblock.

Optional: **`YOLO_MICRO_BATCH_MAX_BUFFER`** — cap how many frames may be held in RAM while waiting.

## Pipeline design (`_run_pipeline_yolo_gated`)

1. **State:** `yolo_pending: list[tuple[int, FrameData]]` kept sorted by `frame_idx`.
2. **Per decoded frame:** Decide commit vs append using the rules above.
3. **On commit:** Single `yolo.detect([...])`, then **in strict index order** for each frame: `gate.observe` → `peeing.update` → same **RF-DETR queue / drain / stash / emit** logic as today (preserve output frame order).
4. **Edge cases:** First frame, last frame, `B=1` (should match current single-frame behavior).

## Config surface

- `settings.py`: `YOLO_MICRO_BATCH` (default could be `8` or `1` to preserve legacy).
- `worker.py`: optional CLI flag mirroring env (same pattern as `YOLO_COARSE_STRIDE`, etc.).
- Extend the existing startup log (strides / streak) with `yolo_micro_batch=...`.

## Validation

1. **Parity / regression:** Short clip, fixed settings; compare `YOLO_MICRO_BATCH=1` vs `N` on outputs that matter (detections, peeing edge events, optional frame hash / diff).
2. **Perf:** Wall time and `yolo_sec` before/after on the same machine and video.

## Acceptance criteria (when implemented)

- **Correctness:** With default gate math, outputs match **batch=1** reference on chosen test clips (within agreed tolerance).
- **Perf:** No regression when `YOLO_MICRO_BATCH=1`.
- **Safety:** Bounded RAM; clear logging for commit reasons and batch sizes (especially EOF tail).

## Optional later (explicit tradeoff)

**Lookahead / speculative dense batching:** Run YOLO on `n, n+2, …` without intermediate `observe` → can **diverge** from current gate if dense would have ended mid-batch. Requires a **separate flag** and user-facing documentation of behavior change.

## Backlog checklist (high level)

- [ ] Document gate semantics vs batching (`observe` order, dense exit, coarse > 1).
- [ ] Finalize `YOLO_MICRO_BATCH` + commit rules (wait-B, must-unblock, EOF tail).
- [ ] Document when max batch applies (coarse 1 vs sparse coarse).
- [ ] Design `_run_pipeline_yolo_gated` state machine (pending queue, per-frame order).
- [ ] Design peeing / RF-DETR / stash interaction after batched `yolo.detect`.
- [ ] Add settings + worker CLI / env wiring (+ startup log).
- [ ] Add tests or scripted parity run on a short clip.
- [ ] (Optional) Speculative dense batching flag + tradeoff doc.

## Related code

- Gate: `worker-python/core/yolo_stride_gate.py`
- Gated pipeline: `worker-python/pipelines/test_pipeline.py` (`_run_pipeline_yolo_gated`)
- YOLO wrapper: `worker-python/models/yolo_detector.py`
- Defaults: `worker-python/settings.py`
