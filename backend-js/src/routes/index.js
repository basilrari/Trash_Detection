// src/routes/index.js
import express from 'express';
import jobsRouter from './jobs.routes.js';

const router = express.Router();

router.use('/jobs', jobsRouter);

// Optional: list all jobs (nice for testing)
router.get('/jobs', async (req, res) => {
  try {
    const jobs = await Job.findAll({
      order: [['createdAt', 'DESC']],
      limit: 50,
    });
    res.json(jobs.map(j => ({
      jobId: j.id,
      status: j.status,
      progress: j.progress,
      createdAt: j.createdAt,
    })));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

export default router;