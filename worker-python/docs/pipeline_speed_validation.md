# Accuracy / behavior validation (after speed tuning)

After each settings or batching change, confirm behavior is still acceptable on your **baseline clip(s)** and a **hard negative** clip if you have one.

## License plates

- [ ] Plate text appears when the plate is clearly visible at moderate range.
- [ ] Locked plate label stays stable when the box jitters slightly (`OCR_LOCK_CONFIDENCE` / location gate).
- [ ] With higher `LP_VEHICLE_LP_STRIDE` or `LP_LOCK_REFRESH_STRIDE`: plate still updates when the vehicle turns or the first read was wrong.

## Vehicles / scene

- [ ] No obvious increase in missed vehicles vs baseline when changing `SCENE_YOLO_TARGET_FRAMES_PER_SECOND` or stride override.
- [ ] `YOLO_CONFIDENCE` changes: check false negative rate on dim/small vehicles.

## RF-DETR / cigarette

- [ ] If `RF_DETR_CIGARETTE_EVERY_N_BATCHES` > 1: spot-check cigarette events still appear within an acceptable delay.

## Peeing

- [ ] Stillness + duration: confirm still behaves on a known positive clip (if applicable).
- [ ] Motorcycle exclusion: confirm riders are not falsely flagged when enabled.

## OCR stability

- [ ] `PADDLE_OCR_ISOLATE_PROCESS=True`: no Paddle/Torch CUDA stream errors during LP-heavy sections.
- [ ] If `OCR_MIN_VARIANCE_LAPLACIAN` > 0: verify you are not losing plates on motion-blur frames that matter for your product.

## When to reject a speed change

Reject or revert a change if any checklist item regresses beyond tolerance, even if wall time improves.
