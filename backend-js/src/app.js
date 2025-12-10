import express from 'express';
import cors from 'cors';
import morgan from 'morgan';
import jobsRouter from './routes/jobs.routes.js';

const app = express();

app.use(cors());
app.use(morgan('dev'));
app.use(express.json());

app.get('/', (req, res) => {
  res.json({ message: 'JS backend is alive!' });
});

app.use('/api/jobs', jobsRouter);

export default app;