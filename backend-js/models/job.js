// models/job.js
import { DataTypes, Model } from 'sequelize';
import sequelize from '../src/db/index.js';
import Sequelize from 'sequelize';

class Job extends Model { }

Job.init(
  {
    id: {
      type: DataTypes.UUID,
      defaultValue: Sequelize.literal('uuid_generate_v4()'),
      primaryKey: true,
    },
    sourceUrl: {
      type: DataTypes.STRING,
      allowNull: false,
    },
    status: {
      type: DataTypes.STRING,
      defaultValue: 'QUEUED',
    },
    progress: {
      type: DataTypes.INTEGER,
      defaultValue: 0,
    },
    outputCsvUrl: DataTypes.STRING,
    errorMessage: DataTypes.TEXT,
    metadata: {
      type: DataTypes.JSONB,
      defaultValue: {},
    },
  },
  {
    sequelize,
    modelName: 'Job',
    tableName: 'Jobs',
  }
);

export default Job;