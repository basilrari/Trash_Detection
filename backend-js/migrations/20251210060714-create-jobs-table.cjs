// migrations/20251210XXXXXX-create-jobs-table.js  (or .cjs — either works)
'use strict';

module.exports = {
  up: async (queryInterface, Sequelize) => {
    await queryInterface.createTable('Jobs', {
      id: {
        type: Sequelize.UUID,
        defaultValue: Sequelize.literal('uuid_generate_v4()'),
        primaryKey: true,
      },
      sourceUrl: {
        type: Sequelize.STRING,
        allowNull: false,
      },
      status: {
        type: Sequelize.STRING,
        defaultValue: 'QUEUED',
      },
      progress: {
        type: Sequelize.INTEGER,
        defaultValue: 0,
      },
      outputCsvUrl: Sequelize.STRING,
      errorMessage: Sequelize.TEXT,
      metadata: {
        type: Sequelize.JSONB,
        defaultValue: {},
      },
      createdAt: {
        type: Sequelize.DATE,
        allowNull: false,
      },
      updatedAt: {
        type: Sequelize.DATE,
        allowNull: false,
      },
    });
  },

  down: async (queryInterface) => {
    await queryInterface.dropTable('Jobs');
  },
};