import React from 'react';
import { useAuth } from '../../../context/AuthContext';

const Navbar = () => {
  const { currentUser, logout } = useAuth();

  const handleLogout = () => {
    logout();
  };

  return (
    <div className="navbar">
      <div className="navbar-left">
        <h2>Tempo</h2>
      </div>
      <div className="navbar-right">
        {currentUser && (
          <>
            <span className="user-greeting">Hello, {currentUser.name}</span>
            <button onClick={handleLogout} className="logout-button">Logout</button>
          </>
        )}
      </div>
    </div>
  );
};

export default Navbar;
