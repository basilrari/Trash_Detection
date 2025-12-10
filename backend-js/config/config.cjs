// config/config.cjs
require('dotenv').config();

const { DATABASE_URL } = process.env;

module.exports = {
  development: {
    url: DATABASE_URL,
    dialect: 'postgres',
    dialectOptions: {
      ssl: { require: true, rejectUnauthorized: false },
    },
  },
  production: {
    url: DATABASE_URL,
    dialect: 'postgres',
    dialectOptions: {
      ssl: { require: true, rejectUnauthorized: false },
    },
  },
};