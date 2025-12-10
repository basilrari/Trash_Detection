// src/routes/jobs.routes.js
import express from 'express';
import Job from '../../models/job.js';
import { enqueueJob } from '../services/queue.service.js';

const router = express.Router();

router.post('/', async (req, res) => {
  try {
    const { sourceUrl, metadata } = req.body;

    if (!sourceUrl) {
      return res.status(400).json({ error: 'sourceUrl is required' });
    }

    const job = await Job.create({
      sourceUrl,
      metadata: metadata || {},
    });

    // This is the fire-and-forget part
    await enqueueJob({
      jobId: job.id,
      sourceUrl: job.sourceUrl,
      metadata: job.metadata,
    });

    res.status(201).json({
      jobId: job.id,
      status: job.status,
      message: 'Job queued for processing',
    });
  } catch (err) {
    console.error('Job creation failed:', err);
    res.status(500).json({ error: err.message });
  }
});

// GET /api/jobs/:id - get job status
router.get('/:id', async (req, res) => {
  try {
    const job = await Job.findByPk(req.params.id);
    if (!job) return res.status(404).json({ error: 'Job not found' });

    res.json({
      jobId: job.id,
      status: job.status,
      progress: job.progress,
      outputCsvUrl: job.outputCsvUrl,
      createdAt: job.createdAt,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

export default router;