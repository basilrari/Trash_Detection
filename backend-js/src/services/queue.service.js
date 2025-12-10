// src/services/queue.service.js
import { Queue } from 'bullmq';
import IORedis from 'ioredis';

const connection = new IORedis(process.env.REDIS_URL || 'redis://localhost:6379');

export const videoQueue = new Queue('videoJobs', {
  connection,
  defaultJobOptions: {
    attempts: 3,
    backoff: { type: 'exponential', delay: 5000 },
    removeOnComplete: true,
    removeOnFail: false,
  },
});

export const enqueueJob = async (jobData) => {
  await videoQueue.add('processVideo', jobData, {
    jobId: jobData.jobId, // prevents duplicates
  });
};