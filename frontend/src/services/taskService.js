import apiClient from './apiClient';

/**
 * Task service for interacting with the tasks API
 */
export const taskService = {
  /**
   * Get all tasks for the current user
   * @param {Object} params - Query parameters
   * @param {number} params.skip - Number of items to skip
   * @param {number} params.limit - Number of items to return
   * @param {string} params.status - Filter by status
   * @returns {Promise} Promise resolving to tasks array
   */
  getTasks: async (params = {}) => {
    try {
      const response = await apiClient.get('/tasks/', { params });
      return response.data;
    } catch (error) {
      console.error('Error fetching tasks:', error);
      throw error;
    }
  },

  /**
   * Get a specific task by ID
   * @param {string} taskId - Task ID
   * @returns {Promise} Promise resolving to task object
   */
  getTask: async (taskId) => {
    try {
      const response = await apiClient.get(`/tasks/${taskId}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching task ${taskId}:`, error);
      throw error;
    }
  },

  /**
   * Create a new task
   * @param {Object} taskData - Task data
   * @returns {Promise} Promise resolving to created task
   */
  createTask: async (taskData) => {
    try {
      const response = await apiClient.post('/tasks/', taskData);
      return response.data;
    } catch (error) {
      console.error('Error creating task:', error);
      throw error;
    }
  },

  /**
   * Update an existing task
   * @param {string} taskId - Task ID
   * @param {Object} taskData - Updated task data
   * @returns {Promise} Promise resolving to updated task
   */
  updateTask: async (taskId, taskData) => {
    try {
      const response = await apiClient.put(`/tasks/${taskId}`, taskData);
      return response.data;
    } catch (error) {
      console.error(`Error updating task ${taskId}:`, error);
      throw error;
    }
  },

  /**
   * Delete a task
   * @param {string} taskId - Task ID
   * @returns {Promise} Promise resolving when task is deleted
   */
  deleteTask: async (taskId) => {
    try {
      await apiClient.delete(`/tasks/${taskId}`);
      return true;
    } catch (error) {
      console.error(`Error deleting task ${taskId}:`, error);
      throw error;
    }
  },

  /**
   * Mark a task as completed
   * @param {string} taskId - Task ID
   * @param {number} actualDuration - Actual duration in minutes
   * @returns {Promise} Promise resolving to updated task
   */
  completeTask: async (taskId, actualDuration) => {
    try {
      const response = await apiClient.post(`/tasks/${taskId}/complete`, { actual_duration: actualDuration });
      return response.data;
    } catch (error) {
      console.error(`Error completing task ${taskId}:`, error);
      throw error;
    }
  }
};

export default taskService;
