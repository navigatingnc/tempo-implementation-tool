import React, { useState } from 'react';
import { useAuth } from '../../context/AuthContext';

const SettingsPage = () => {
  const { currentUser } = useAuth();
  const [generalSettings, setGeneralSettings] = useState({
    theme: 'light',
    notifications: true,
    focusMode: 'pomodoro',
    defaultFocusDuration: 25
  });
  const [integrations, setIntegrations] = useState([
    { id: 'google', name: 'Google Calendar', connected: false },
    { id: 'outlook', name: 'Outlook Calendar', connected: false },
    { id: 'slack', name: 'Slack', connected: false },
    { id: 'github', name: 'GitHub', connected: false }
  ]);
  
  const handleGeneralSettingChange = (setting, value) => {
    setGeneralSettings(prev => ({
      ...prev,
      [setting]: value
    }));
  };
  
  const toggleIntegration = (id) => {
    setIntegrations(prev => 
      prev.map(integration => 
        integration.id === id 
          ? { ...integration, connected: !integration.connected } 
          : integration
      )
    );
  };
  
  return (
    <div className="settings-page">
      <h1>Settings</h1>
      
      <div className="settings-section">
        <h2>General Settings</h2>
        <div className="settings-form">
          <div className="form-group">
            <label htmlFor="theme">Theme:</label>
            <select 
              id="theme" 
              value={generalSettings.theme}
              onChange={(e) => handleGeneralSettingChange('theme', e.target.value)}
            >
              <option value="light">Light</option>
              <option value="dark">Dark</option>
              <option value="system">System Default</option>
            </select>
          </div>
          
          <div className="form-group">
            <label htmlFor="notifications">
              <input 
                type="checkbox" 
                id="notifications" 
                checked={generalSettings.notifications}
                onChange={(e) => handleGeneralSettingChange('notifications', e.target.checked)}
              />
              Enable Notifications
            </label>
          </div>
          
          <div className="form-group">
            <label htmlFor="focusMode">Focus Mode:</label>
            <select 
              id="focusMode" 
              value={generalSettings.focusMode}
              onChange={(e) => handleGeneralSettingChange('focusMode', e.target.value)}
            >
              <option value="pomodoro">Pomodoro</option>
              <option value="flowtime">Flowtime</option>
              <option value="custom">Custom</option>
            </select>
          </div>
          
          <div className="form-group">
            <label htmlFor="defaultFocusDuration">Default Focus Duration (minutes):</label>
            <input 
              type="number" 
              id="defaultFocusDuration" 
              value={generalSettings.defaultFocusDuration}
              onChange={(e) => handleGeneralSettingChange('defaultFocusDuration', parseInt(e.target.value))}
              min="1"
              max="120"
            />
          </div>
        </div>
      </div>
      
      <div className="settings-section">
        <h2>Integrations</h2>
        <div className="integrations-list">
          {integrations.map(integration => (
            <div key={integration.id} className="integration-item">
              <span className="integration-name">{integration.name}</span>
              <button 
                className={`integration-toggle ${integration.connected ? 'connected' : ''}`}
                onClick={() => toggleIntegration(integration.id)}
              >
                {integration.connected ? 'Disconnect' : 'Connect'}
              </button>
            </div>
          ))}
        </div>
      </div>
      
      <div className="settings-section">
        <h2>Account</h2>
        <div className="account-info">
          <p><strong>Email:</strong> {currentUser?.email}</p>
          <p><strong>Name:</strong> {currentUser?.name}</p>
          <div className="account-actions">
            <button className="change-password-button">Change Password</button>
            <button className="export-data-button">Export My Data</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsPage;
