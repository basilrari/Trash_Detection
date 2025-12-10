ai-video-platform/
  backend-js/              # Node.js API (jobs, status, etc.)
    src/
      app.js               # Create Express app, mount routes/middlewares
      server.js            # Start HTTP server (reads PORT, etc.)

      routes/
        index.js           # Combine and mount all route modules on /api
        jobs.routes.js     # Define /api/jobs, /api/jobs/:id, /api/jobs list

      controllers/
        jobs.controller.js # HTTP handlers: parse req, call services, send res

      services/
        jobs.service.js    # Core job logic: create job, get job, list jobs
        queue.service.js   # Enqueue job messages to Redis/Rabbit/etc.
        db.service.js      # DB access helpers (wraps models/queries)

      models/
        job.model.js       # Job schema + queries (using chosen DB library)

      middlewares/
        auth.middleware.js # Auth / user extraction (JWT, API key, etc.)
        error.middleware.js# Central error handler for Express
        validate.middleware.js # Request validation (Joi/Zod/custom)

      utils/
        logger.js          # Logging wrapper (console/winston/pino)
        pagination.js      # Helpers for paginated job listing
        config.js          # Read env vars, config constants

    package.json           # Dependencies, scripts
    .env.example           # Example env vars (DB_URL, REDIS_URL, etc.)

  worker-python/           # Python GPU worker service
    your_project/          # Python package root (rename later if you like)
      __init__.py

      config/
        __init__.py
        settings.py        # Paths, env loading, thresholds, model configs

      core/
        __init__.py
        types.py           # Dataclasses for FrameData, Detection, etc.
        video_reader.py    # Generic video/stream reader (chunked)
        writer.py          # CSV writer / result writer
        utils.py           # Shared helpers (timing, batching, etc.)

      models/              # ML model wrappers (each with clear interface)
        __init__.py
        base.py            # Abstract base classes: Detector, OCRModel, etc.
        yolo_detector.py   # Person/vehicle detection wrapper
        lp_detector.py     # License plate detector wrapper
        ocr.py             # PaddleOCR wrapper
        pose.py            # MediaPipe pose estimation wrapper
        trash_detector.py  # RFDETR trash detection wrapper

      pipelines/           # High-level processing flows
        __init__.py
        base.py            # Pipeline interface (process_video, etc.)
        traffic_pipeline.py# Person/vehicle + license plate + OCR flow
        behavior_pipeline.py # Pose/urination detection flow
        litter_pipeline.py # Trash detection (RFDETR) flow

      jobs/
        __init__.py
        processor.py       # JobProcessor: given job payload, runs pipeline(s)
                           # - Fetch video from Dropbox/Drive
                           # - Use video_reader + pipelines
                           # - Write CSV via writer
                           # - Update job status via DB/API

      services/
        __init__.py
        queue_consumer.py  # Connect to queue, pull jobs, call JobProcessor
        db_client.py       # Optional: DB client used by worker (if direct)
        status_client.py   # Optional: HTTP client to update backend-js

      scripts/
        run_worker.py      # CLI entrypoint: load models, start consumer
        debug_video.py     # Local dev: run pipeline on one video, no queue

    requirements.txt       # Python deps (torch, opencv, redis, requests, etc.)

  .gitignore
  README.md
