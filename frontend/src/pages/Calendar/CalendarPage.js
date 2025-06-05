import React, { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';

const CalendarPage = () => {
  const { currentUser } = useAuth();
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentDate, setCurrentDate] = useState(new Date());
  
  useEffect(() => {
    // This would normally fetch calendar events from the API
    const fetchEvents = async () => {
      try {
        setLoading(true);
        // Mock data for demonstration
        const mockEvents = [
          { 
            id: '1', 
            title: 'Daily standup', 
            start_time: '2025-06-05T09:30:00Z', 
            end_time: '2025-06-05T10:00:00Z',
            location: 'Conference Room A'
          },
          { 
            id: '2', 
            title: 'Project planning', 
            start_time: '2025-06-05T11:00:00Z', 
            end_time: '2025-06-05T12:30:00Z',
            location: 'Virtual Meeting'
          },
          { 
            id: '3', 
            title: 'Client presentation', 
            start_time: '2025-06-06T14:00:00Z', 
            end_time: '2025-06-06T15:30:00Z',
            location: 'Main Office'
          }
        ];
        
        setEvents(mockEvents);
      } catch (err) {
        console.error('Error fetching calendar events:', err);
        setError('Failed to load calendar events. Please try again later.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchEvents();
  }, []);
  
  const navigateMonth = (direction) => {
    const newDate = new Date(currentDate);
    newDate.setMonth(newDate.getMonth() + direction);
    setCurrentDate(newDate);
  };
  
  return (
    <div className="calendar-page">
      <h1>Calendar</h1>
      
      <div className="calendar-actions">
        <button className="add-event-button">Add New Event</button>
        <div className="calendar-navigation">
          <button onClick={() => navigateMonth(-1)}>Previous Month</button>
          <h2>{currentDate.toLocaleString('default', { month: 'long', year: 'numeric' })}</h2>
          <button onClick={() => navigateMonth(1)}>Next Month</button>
        </div>
      </div>
      
      {loading ? (
        <p>Loading calendar...</p>
      ) : error ? (
        <p className="error-message">{error}</p>
      ) : (
        <div className="calendar-container">
          <div className="calendar-grid">
            <div className="calendar-header">
              {['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'].map(day => (
                <div key={day} className="calendar-day-header">{day}</div>
              ))}
            </div>
            <div className="calendar-body">
              <p>Calendar grid would be rendered here</p>
            </div>
          </div>
          
          <div className="events-list">
            <h3>Upcoming Events</h3>
            {events.length === 0 ? (
              <p>No events scheduled. Add your first event to get started!</p>
            ) : (
              events.map(event => (
                <div key={event.id} className="event-item">
                  <div className="event-time">
                    {new Date(event.start_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    {' - '}
                    {new Date(event.end_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                  <div className="event-details">
                    <h4>{event.title}</h4>
                    <p>{event.location}</p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default CalendarPage;
