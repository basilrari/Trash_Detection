Trash Detection, Behavior Detection & License Plate Analysis System

A modular, GPU-accelerated, multi-model video analysis pipeline

This repository contains a two-service system for analyzing long CCTV-style videos using multiple computer-vision models.
The design is intentionally LLM-friendly: folders, modules, and abstractions are structured so models and pipelines can be swapped or extended safely.

1. High-Level Overview

The system analyzes long videos (hours, even full-day CCTV files) by streaming them in chunks and applying a sequence of detectors and classifiers.
It answers questions like:

Were there people or vehicles present?

Which vehicles were seen, and what license plates did they have?

Did any person urinate in the camera’s view (pose + custom logic)?

Was any trash/litter present?

What is the timeline of these events?

Output is a single CSV file per video, containing timestamps, detections, and all associated metadata.

The stack consists of:

JS Backend (Express)

Receives video-processing requests

Creates job records

Dispatches jobs to a queue

Exposes APIs to fetch job status + results

Python Worker (GPU)

Streams video

Runs YOLO-based activity gating

Runs license-plate detector + OCR

Runs pose estimation for behavior analysis

Runs RFDETR for litter detection

Writes unified CSV results

Updates backend with job progress

The worker uses CUDA, PyTorch, PaddleOCR, MediaPipe, RFDETR, and custom models.

2. System Architecture
Client
  ↓
backend-js (Express API)
  - Receives video link
  - Creates job record (DB)
  - Pushes job to queue
  ↓
Queue  (Redis / SQS / etc.)
  ↓
worker-python (GPU)
  - Downloads/streams video in chunks
  - YOLO → LP detector → OCR
  - Pose estimation (urination detection)
  - Litter detector (RFDETR)
  - Writes CSV
  - Uploads CSV / updates job
  ↓
backend-js
  - exposes /api/jobs/:id for results


The Python worker performs all GPU tasks; the JS backend handles HTTP + DB + queue coordination.

3. Folder Structure Overview
📁 backend-js/
  ├── src/
  │   ├── app.js                # Express composition root
  │   ├── server.js             # Starts HTTP server
  │   ├── routes/               # API routes (/api/jobs)
  │   ├── controllers/          # Job controller logic
  │   ├── services/             # queue.service, db access
  │   └── db/                   # DB connection/config
  ├── models/                   # job model schema + queries
  ├── config/                   # env, database configs
  ├── migrations/               # DB migrations
  └── package.json

📁 worker-python/
  ├── worker.py                 # Worker entrypoint (starts queue consumer)
  ├── settings.py               # Configurable paths, weights, thresholds
  ├── services/
  │   └── queue_consumer.py     # Pulls jobs from queue, launches job processor
  ├── jobs/
  │   └── processor.py          # Orchestrates full video analysis for one job
  ├── pipelines/
  │   ├── test_pipeline.py      # Manual test runner
  │   └── ...                   # future: traffic/behavior/litter pipelines
  ├── models/
  │   ├── yolo_detector.py      # People/vehicle detector (YOLO)
  │   ├── lp_detector.py        # License plate detector
  │   ├── ocr.py                # PaddleOCR wrapper (GPU/CPU fallbacks)
  │   ├── base.py               # Model interfaces (Detector, OCRModel, etc.)
  │   └── ...
  ├── core/
  │   ├── types.py              # Structured entities (FrameData, Detection…)
  ├── weights/                  # yolo11x.pt, bestlicense.pt
  └── output_with_boxes.mp4     # sample output


This layout is intentionally stable so an LLM or developer can navigate and extend safely.

4. Video Processing Pipeline (Python)

The worker processes each job using a chunked streaming pipeline:

1. Video Streaming

The worker never loads a long video fully.
Chunks may be sequential frame batches or time-sliced windows.

2. Activity Gating (YOLO)

YOLO detects people and vehicles.
If no activity, the chunk is skipped to save GPU time.

3. License Plate Processing (only for vehicle chunks)

Custom license plate detector isolates plate regions

Each plate crop is passed to PaddleOCR

OCR returns the best text candidate (robust fallback logic included)

4. Behavior Analysis (only for chunks with people/vehicles)

Pose estimation (MediaPipe)

Custom event detection (e.g., urination logic)

5. Litter Detection (only for relevant chunks)

RFDETR detects trash/litter around subjects

6. CSV Output

The worker aggregates everything into one CSV per job:

Columns may include:

timestamp

chunk frame indices

bounding boxes

labels (person/vehicle/plate/trash)

OCR result

pose analysis result

confidence scores

event identifiers (e.g., urination event)

5. Backend-JS Responsibilities

POST /api/jobs → create job, queue it

GET /api/jobs/:id → return job status & result CSV URL

Maintains DB records for:

QUEUED

PROCESSING

DONE

FAILED

Queue service provides enqueue(jobPayload)

DB service provides basic CRUD helpers

Clean separation of controllers vs services vs models

6. Communication Contract

The JS backend sends a job payload to the queue:

{
  jobId: "...",
  videoUrl: "...",
  cameraId: "...",
  date: "...",
  options: { ... }
}


The Python worker must:

Set status → PROCESSING

Stream+analyze video

Produce CSV

Upload CSV or save locally

Set status → DONE with outputCsvUrl

On errors, set status → FAILED with errorMessage

These contracts are intentionally simple so they can be expanded easily.

7. Running the System
1. Start JS backend
cd backend-js
npm install
npm run start  # or node src/server.js

2. Start Python worker (GPU environment)
conda activate paddle-gpu
python worker-python/worker.py

3. Run test pipeline manually
python worker-python/pipelines/test_pipeline.py

4. Create a job (example)
curl -X POST http://localhost:3000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{"videoUrl":"https://dropbox.com/yourvideo.mp4"}'

8. Model Weights and GPU Notes

YOLO: worker-python/weights/yolo11x.pt

License Plates: worker-python/weights/bestlicense.pt

PaddleOCR automatically selects GPU if device="gpu"

PyTorch must show cuda_available: True in logs

Worker prints GPU diagnostics at startup (if enabled)