This system processes long CCTV-style videos with multiple computer vision tasks in a **single, coordinated pipeline**, while remaining modular so individual models can be swapped without breaking the overall flow. The codebase is split into a JS API service and a Python GPU worker, connected via a job + queue mechanism, and the architecture description is written so an LLM can reason about structure, responsibilities, and contracts.[1][2][3]

## High-level behavior of the system

For each input video (e.g., 24-hour CCTV footage hosted on Dropbox/Drive), the system treats it as **one job**. That job is created by the JS backend and executed by the Python worker. The Python worker runs a **full analysis pipeline**:

1. **Stream the video in chunks** (e.g., by time window or frame batch) instead of loading everything at once.[4][5]
2. On each chunk, run **YOLO** to detect persons and vehicles. This acts as an “activity gate”:  
   - If no people or vehicles are found in a chunk, the system records that nothing relevant happened for that time and **skips heavier models** for that chunk.  
   - If people or vehicles are detected, the system marks that chunk as interesting and then runs additional analyses.  
3. For **chunks with activity**, the worker runs:  
   - License plate detection on vehicle regions (custom fine‑tuned model).  
   - **OCR (PaddleOCR)** on license plate crops to extract plate text.  
   - **Pose estimation (MediaPipe)** and custom logic to detect urination events around those persons/vehicles.  
   - **RFDETR** to detect trash/litter in the same region/time window.  
4. All results from these steps are aggregated into a **single CSV per video**. Each row describes an event or detection (e.g., vehicle sighting with plate, urination event, litter detection), with a schema that includes timestamp, location/camera, type of event, bounding boxes, confidence, and any extra data (like plate text).  

This design ensures the GPU-heavy models are only used where there is likely to be relevant activity, while still giving you a unified output file per video.  

## JS backend (backend-js) – intent and structure

The JS backend is responsible for **receiving job requests, tracking job state, and exposing HTTP APIs**. It does not run any GPU models itself.[3][6]

Key responsibilities:  

- Accept links or IDs of long videos (Dropbox/Drive) from clients.  
- Create a **job record** in a database with status, timestamps, and references to input and output.  
- Send a **job message** to a queue that the Python worker consumes.  
- Provide endpoints for clients to check job status and fetch the CSV result.  

Important folders and their intent:  

- `backend-js/src/app.js`: Builds the Express app, attaches JSON parsing, CORS, and mounts routers under `/api`. This is the composition root for routes and middlewares.  
- `backend-js/src/server.js`: Starts the HTTP server with a configurable port and environment; this is the only file that calls `app.listen`.  
- `routes/`:  
  - `routes/index.js`: Mounts feature-specific routers, such as `/api/jobs`.  
  - `routes/jobs.routes.js`: Defines endpoints like  
    - `POST /api/jobs` (create a video-processing job), and  
    - `GET /api/jobs/:id` (retrieve job status and output CSV URL).[7]
- `controllers/`:  
  - `jobs.controller.js`: Translates HTTP requests to service calls. It reads the video link and other metadata from the request, calls `jobs.service.createJob`, then returns a response containing a `jobId`. It also handles reading a job by ID.  
- `services/`:  
  - `jobs.service.js`: Core job lifecycle logic. It validates inputs, creates job records through models/DB helpers, and calls `queue.service.enqueue(jobPayload)` to send a message to the worker queue. It also provides methods to fetch jobs and update job status if needed.  
  - `queue.service.js`: Abstracts away the concrete queue implementation (e.g., Redis + Bull/bee-queue, RabbitMQ, or SQS). It exposes functions like `enqueue(jobPayload)` so the rest of the backend does not depend on the specific queue library.[8][9]
  - `db.service.js`: Manages database connections and higher-level helpers for executing queries or transactions.  
- `models/`:  
  - `job.model.js`: Encapsulates the job table and DB queries. A job record contains fields like `id`, `inputSourceType` (Dropbox/Drive), `inputSourceIdOrUrl`, `status` (QUEUED/PROCESSING/DONE/FAILED), `progress`, `outputCsvUrl`, `errorMessage`, timestamps, and possibly metadata like `cameraId` and `date`.  
- `middlewares/` and `utils/`:  
  - `auth.middleware.js`: Optionally attach user identity to requests (if you have an authenticated multi‑user system).  
  - `error.middleware.js`: Centralized HTTP error handling.  
  - `validate.middleware.js`: Request validation using a schema library.  
  - `utils/logger.js`: Logging abstraction.  
  - `utils/config.js`: Loads environment variables and config constants (DB URL, queue URL, etc.).[2][1]

The intent is to keep HTTP concerns, business logic, persistence, and infrastructure cleanly separated so that the JS code remains easy to extend and reason about.  

## Python worker (worker-python) – intent and structure

The Python worker is a **long‑running GPU service** that reads job messages from a queue, fetches video content, runs the full multi-model analysis, and writes results back (CSV + status updates).[10][4]

Key responsibilities:  

- Consume jobs from a message queue.  
- For each job:  
  - Download/stream video from Dropbox/Drive in chunks.  
  - Run the conditional YOLO‑gated pipeline (traffic + behavior + litter).  
  - Write a single CSV per input video.  
  - Update job status and progress in the database or via the JS API.  

Important package areas and their intent:  

- `worker/config/settings.py`: Central configuration (e.g., queue endpoints, DB/API URLs, model paths, thresholds for detection, chunk sizes).  
- `worker/core/`:  
  - `types.py`: Defines structured types like `FrameData` (frame index, timestamp, image data reference), `Detection` (bbox, label, confidence), `LicensePlate`, `PoseResult`, and other common entities used across models and pipelines.  
  - `video_reader.py`: Responsible for streaming or chunked reading of videos from a URL or storage ID (Dropbox/Drive). It hides I/O details and ensures that a 24‑hour video is processed in memory-safe segments.[5][4]
  - `writer.py`: Handles CSV writing and potentially uploading the final CSV to object storage. It provides consistent formatting so pipelines just send structured data to it.  
  - `utils.py`: General utilities (timing, GPU device selection, batching helpers, error wrappers).  
- `worker/models/`:  
  - `base.py`: Defines abstract interfaces for model wrappers, such as:  
    - `Detector.detect(frames)` for YOLO‑style detectors (returning persons/vehicles).  
    - `LicensePlateDetector.detect_plates(frames)` for plate bounding boxes.  
    - `OCRModel.recognize(crops)` for text extraction from plate crops.  
    - `PoseEstimator.estimate(frames)` for pose keypoints.  
    - `TrashDetector.detect_trash(frames)` for RFDETR‑style trash detection.  
    The intent is that pipelines depend only on these interfaces, not on YOLO, PaddleOCR, or RFDETR directly.  
  - `yolo_detector.py`: Implements `Detector` using YOLO for people and vehicles.  
  - `lp_detector.py`: Implements `LicensePlateDetector` using the custom fine‑tuned plate model.  
  - `ocr.py`: Implements `OCRModel` via PaddleOCR.  
  - `pose.py`: Implements `PoseEstimator` using MediaPipe.  
  - `trash_detector.py`: Implements `TrashDetector` using RFDETR.  
- `worker/pipelines/`:  
  - `base.py`: Defines a general `Pipeline` interface with methods like `process_video(video_iterable, writer)` which accept streamed frames and use `writer` to persist results.  
  - `traffic_pipeline.py`: Handles the first stage of the flow. For each video chunk, it:  
    - Runs YOLO to find people and vehicles.  
    - On vehicles, runs license plate detection and OCR.  
    - Marks chunks as “activity present” or not and forwards relevant information to the coordinating entity (either the JobProcessor or a higher‑level pipeline).  
    - Writes traffic‑related detections (vehicles, plates) into the CSV via `writer`.  
  - `behavior_pipeline.py`: For frames/chunks where YOLO found people/vehicles, it runs pose estimation and your urination-logic to detect behavior events, and appends those entries to the same CSV.  
  - `litter_pipeline.py`: For those same “interesting” chunks, it runs RFDETR to detect trash and appends litter events to the CSV.  

In practice, you can wrap these three as a single `FullAnalysisPipeline` or have the `JobProcessor` orchestrate them in sequence over the same stream of frames. The key intent is that:  

- **YOLO is the gate**: it decides where to run more expensive models.  
- All results from traffic, behavior, and litter analysis end up in **one CSV** per job.  

- `worker/jobs/processor.py`:  
  - Contains `JobProcessor`, which is the core orchestrator for one job.  
  - Given a job payload (`jobId`, video link, metadata), it:  
    - Updates job status to `PROCESSING`.  
    - Uses `video_reader` to iterate over the video in chunks.  
    - For each chunk, calls the traffic pipeline (YOLO + LP + OCR) to get detections.  
    - For chunks with detected persons/vehicles, calls behavior and litter pipelines on the same frames or a subset.  
    - Uses `writer` to append all events into a single CSV file.  
    - When finished, uploads CSV if needed, updates job status to `DONE` and sets `outputCsvUrl`.  
    - On error, logs and sets job status to `FAILED` with an error message.  
- `worker/services/queue_consumer.py`:  
  - The long-running loop that connects to the queue, pulls job messages, and invokes `JobProcessor` for each. It may handle retries and backoff.[3]
- `worker/services/db_client.py` / `status_client.py`:  
  - DB or HTTP client logic for updating job status and progress in the JS backend or directly in the database.  
- `worker/scripts/run_worker.py`:  
  - CLI entry point to start a worker process. It loads settings, initializes model wrappers (YOLO, LP detector, OCR, pose, RFDETR) once, constructs pipelines and a `JobProcessor` factory, and starts `queue_consumer`.  

## Contract and communication intent

The JS and Python sides communicate through **clear contracts** documented under `docs/`:

- `api-contracts.md`: defines REST shapes (for `POST /api/jobs`, `GET /api/jobs/:id`, etc.), so both frontend and backend agree on JSON.  
- `queue-schema.md`: defines the job message format sent from JS to Python (fields like `jobId`, `sourceType`, `sourceIdOrUrl`, `metadata`) and the meaning of each status (QUEUED, PROCESSING, DONE, FAILED).[3]
- `architecture.md`: explains the full flow (client → backend-js → DB/queue → worker-python → CSV/storage → backend-js) using diagrams and text for humans and LLMs.  

The intent is that any tool or LLM reading this repo can infer:  

- Where to add or modify endpoints (in `backend-js/routes` and `controllers`).  
- Where to add or swap models (in `worker-python/worker/models`).  
- Where to change how different analyses are combined (in `pipelines` and `jobs/processor.py`).  
- How to safely extend the system with new event types or new models without breaking existing contracts.  