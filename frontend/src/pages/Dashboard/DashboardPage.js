import React from 'react';
import { useAuth } from '../../context/AuthContext';

const DashboardPage = () => {
  const { currentUser } = useAuth();

  return (
    <div className="dashboard-page">
      <h1>Dashboard</h1>
      <div className="dashboard-welcome">
        <h2>Welcome back, {currentUser?.name || 'User'}!</h2>
        <p>Here's your productivity overview for today.</p>
      </div>
      
      <div className="dashboard-metrics">
        <div className="metric-card">
          <h3>Focus Score</h3>
          <div className="metric-value">7.5</div>
          <div className="metric-trend positive">+0.5 from yesterday</div>
        </div>
        
        <div className="metric-card">
          <h3>Tasks Completed</h3>
          <div className="metric-value">3/8</div>
          <div className="metric-trend">38% completion rate</div>
        </div>
        
        <div className="metric-card">
          <h3>Focus Time</h3>
          <div className="metric-value">2h 15m</div>
          <div className="metric-trend negative">-30m from average</div>
        </div>
      </div>
      
      <div className="dashboard-sections">
        <div className="dashboard-section">
          <h3>Today's Schedule</h3>
          <p>Your schedule will appear here.</p>
        </div>
        
        <div className="dashboard-section">
          <h3>Priority Tasks</h3>
          <p>Your priority tasks will appear here.</p>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
