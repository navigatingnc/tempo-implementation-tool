import React, { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';

const TasksPage = () => {
  const { currentUser } = useAuth();
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    // This would normally fetch tasks from the API
    const fetchTasks = async () => {
      try {
        setLoading(true);
        // Mock data for demonstration
        const mockTasks = [
          { id: '1', title: 'Complete project proposal', status: 'pending', priority: 1, due_date: '2025-06-10T00:00:00Z' },
          { id: '2', title: 'Review code changes', status: 'in_progress', priority: 2, due_date: '2025-06-07T00:00:00Z' },
          { id: '3', title: 'Team meeting preparation', status: 'pending', priority: 1, due_date: '2025-06-05T16:00:00Z' },
          { id: '4', title: 'Update documentation', status: 'completed', priority: 3, due_date: '2025-06-03T00:00:00Z' }
        ];
        
        setTasks(mockTasks);
      } catch (err) {
        console.error('Error fetching tasks:', err);
        setError('Failed to load tasks. Please try again later.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchTasks();
  }, []);
  
  return (
    <div className="tasks-page">
      <h1>Tasks</h1>
      
      <div className="tasks-actions">
        <button className="add-task-button">Add New Task</button>
        <div className="tasks-filters">
          <select defaultValue="all">
            <option value="all">All Tasks</option>
            <option value="pending">Pending</option>
            <option value="in_progress">In Progress</option>
            <option value="completed">Completed</option>
          </select>
          <select defaultValue="due_date">
            <option value="due_date">Sort by Due Date</option>
            <option value="priority">Sort by Priority</option>
            <option value="title">Sort by Title</option>
          </select>
        </div>
      </div>
      
      {loading ? (
        <p>Loading tasks...</p>
      ) : error ? (
        <p className="error-message">{error}</p>
      ) : (
        <div className="tasks-list">
          {tasks.length === 0 ? (
            <p>No tasks found. Create your first task to get started!</p>
          ) : (
            tasks.map(task => (
              <div key={task.id} className={`task-item status-${task.status}`}>
                <div className="task-header">
                  <h3>{task.title}</h3>
                  <span className={`priority priority-${task.priority}`}>
                    Priority: {task.priority}
                  </span>
                </div>
                <div className="task-details">
                  <span className="task-status">{task.status}</span>
                  <span className="task-due-date">
                    Due: {new Date(task.due_date).toLocaleDateString()}
                  </span>
                </div>
                <div className="task-actions">
                  <button>Edit</button>
                  <button>Complete</button>
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
};

export default TasksPage;
