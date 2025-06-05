import React from 'react';
import { NavLink } from 'react-router-dom';
import { useAuth } from '../../../context/AuthContext';

const Sidebar = () => {
  const { currentUser } = useAuth();

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h3>Tempo</h3>
        <p>Productivity & Time Management</p>
      </div>
      <div className="sidebar-user">
        {currentUser && (
          <div className="user-info">
            <span className="user-name">{currentUser.name}</span>
            <span className="user-email">{currentUser.email}</span>
          </div>
        )}
      </div>
      <nav className="sidebar-nav">
        <ul>
          <li>
            <NavLink to="/" end className={({ isActive }) => isActive ? 'active' : ''}>
              Dashboard
            </NavLink>
          </li>
          <li>
            <NavLink to="/tasks" className={({ isActive }) => isActive ? 'active' : ''}>
              Tasks
            </NavLink>
          </li>
          <li>
            <NavLink to="/calendar" className={({ isActive }) => isActive ? 'active' : ''}>
              Calendar
            </NavLink>
          </li>
          <li>
            <NavLink to="/analytics" className={({ isActive }) => isActive ? 'active' : ''}>
              Analytics
            </NavLink>
          </li>
          <li>
            <NavLink to="/settings" className={({ isActive }) => isActive ? 'active' : ''}>
              Settings
            </NavLink>
          </li>
        </ul>
      </nav>
    </div>
  );
};

export default Sidebar;
