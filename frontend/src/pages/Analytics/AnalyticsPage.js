import React, { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';

const AnalyticsPage = () => {
  const { currentUser } = useAuth();
  const [analyticsData, setAnalyticsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeRange, setTimeRange] = useState('week');
  
  useEffect(() => {
    // This would normally fetch analytics data from the API
    const fetchAnalytics = async () => {
      try {
        setLoading(true);
        // Mock data for demonstration
        const mockData = {
          productivity_metrics: {
            focus_score: 7.5,
            tasks_completed: 15,
            focus_time_minutes: 480,
            average_session_length: 25,
            interruption_rate: 0.2,
            completion_rate: 0.85
          },
          time_allocation: {
            categories: {
              development: 240,
              meetings: 180,
              planning: 60,
              email: 45,
              breaks: 30,
              other: 60
            },
            total_tracked_minutes: 615
          },
          task_stats: {
            completed_on_time: 12,
            completed_late: 3,
            not_completed: 2,
            average_completion_time: 85,
            estimated_vs_actual_ratio: 1.2
          },
          focus_stats: {
            total_sessions: 18,
            total_focus_time: 480,
            average_session_length: 25,
            completion_rate: 0.9,
            interruption_rate: 0.15,
            average_productivity_score: 8.2
          }
        };
        
        setAnalyticsData(mockData);
      } catch (err) {
        console.error('Error fetching analytics data:', err);
        setError('Failed to load analytics data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchAnalytics();
  }, [timeRange]);
  
  const handleTimeRangeChange = (e) => {
    setTimeRange(e.target.value);
  };
  
  return (
    <div className="analytics-page">
      <h1>Analytics</h1>
      
      <div className="analytics-controls">
        <div className="time-range-selector">
          <label htmlFor="time-range">Time Range:</label>
          <select 
            id="time-range" 
            value={timeRange} 
            onChange={handleTimeRangeChange}
          >
            <option value="day">Today</option>
            <option value="week">This Week</option>
            <option value="month">This Month</option>
            <option value="quarter">This Quarter</option>
            <option value="year">This Year</option>
          </select>
        </div>
      </div>
      
      {loading ? (
        <p>Loading analytics data...</p>
      ) : error ? (
        <p className="error-message">{error}</p>
      ) : analyticsData ? (
        <div className="analytics-dashboard">
          <div className="analytics-section">
            <h2>Productivity Overview</h2>
            <div className="metrics-grid">
              <div className="metric-card">
                <h3>Focus Score</h3>
                <div className="metric-value">{analyticsData.productivity_metrics.focus_score}</div>
              </div>
              <div className="metric-card">
                <h3>Tasks Completed</h3>
                <div className="metric-value">{analyticsData.productivity_metrics.tasks_completed}</div>
              </div>
              <div className="metric-card">
                <h3>Focus Time</h3>
                <div className="metric-value">{Math.floor(analyticsData.productivity_metrics.focus_time_minutes / 60)}h {analyticsData.productivity_metrics.focus_time_minutes % 60}m</div>
              </div>
              <div className="metric-card">
                <h3>Completion Rate</h3>
                <div className="metric-value">{Math.round(analyticsData.productivity_metrics.completion_rate * 100)}%</div>
              </div>
            </div>
          </div>
          
          <div className="analytics-section">
            <h2>Time Allocation</h2>
            <div className="chart-placeholder">
              <p>Time allocation chart would be rendered here</p>
              <ul>
                {Object.entries(analyticsData.time_allocation.categories).map(([category, minutes]) => (
                  <li key={category}>
                    {category}: {Math.floor(minutes / 60)}h {minutes % 60}m ({Math.round((minutes / analyticsData.time_allocation.total_tracked_minutes) * 100)}%)
                  </li>
                ))}
              </ul>
            </div>
          </div>
          
          <div className="analytics-section">
            <h2>Task Completion</h2>
            <div className="chart-placeholder">
              <p>Task completion chart would be rendered here</p>
              <ul>
                <li>Completed on time: {analyticsData.task_stats.completed_on_time}</li>
                <li>Completed late: {analyticsData.task_stats.completed_late}</li>
                <li>Not completed: {analyticsData.task_stats.not_completed}</li>
              </ul>
            </div>
          </div>
          
          <div className="analytics-section">
            <h2>Focus Sessions</h2>
            <div className="chart-placeholder">
              <p>Focus session chart would be rendered here</p>
              <ul>
                <li>Total sessions: {analyticsData.focus_stats.total_sessions}</li>
                <li>Average length: {analyticsData.focus_stats.average_session_length} minutes</li>
                <li>Interruption rate: {Math.round(analyticsData.focus_stats.interruption_rate * 100)}%</li>
              </ul>
            </div>
          </div>
        </div>
      ) : (
        <p>No analytics data available.</p>
      )}
    </div>
  );
};

export default AnalyticsPage;
