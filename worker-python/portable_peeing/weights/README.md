# Weights (pose only)

Person boxes come from **your** detector (e.g. DFINE) — see `../DFINE_INTEGRATION.md`.

| File | Use |
|------|-----|
| `yolo11n-pose.pt` | Default for Basil_Test (`models/yolo_pose/yolo11n-pose.pt` in main repo) |
| `yolo11n-pose_b8_fp16.engine` | Optional TensorRT export later (`--pose-model`) |
