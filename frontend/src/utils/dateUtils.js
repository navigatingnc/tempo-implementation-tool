// Utility functions for date formatting and manipulation

/**
 * Format a date to a human-readable string
 * @param {Date|string} date - Date object or ISO string
 * @param {string} format - Format type ('short', 'long', 'time', 'datetime')
 * @returns {string} Formatted date string
 */
export const formatDate = (date, format = 'short') => {
  if (!date) return '';
  
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  
  switch (format) {
    case 'short':
      return dateObj.toLocaleDateString();
    case 'long':
      return dateObj.toLocaleDateString(undefined, { 
        weekday: 'long', 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric' 
      });
    case 'time':
      return dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    case 'datetime':
      return `${dateObj.toLocaleDateString()} ${dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    default:
      return dateObj.toLocaleDateString();
  }
};

/**
 * Check if a date is today
 * @param {Date|string} date - Date object or ISO string
 * @returns {boolean} True if date is today
 */
export const isToday = (date) => {
  if (!date) return false;
  
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  const today = new Date();
  
  return dateObj.getDate() === today.getDate() &&
    dateObj.getMonth() === today.getMonth() &&
    dateObj.getFullYear() === today.getFullYear();
};

/**
 * Get relative time description (e.g., "2 days ago", "in 3 hours")
 * @param {Date|string} date - Date object or ISO string
 * @returns {string} Relative time description
 */
export const getRelativeTimeDescription = (date) => {
  if (!date) return '';
  
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  const now = new Date();
  const diffMs = dateObj - now;
  const diffSec = Math.round(diffMs / 1000);
  const diffMin = Math.round(diffSec / 60);
  const diffHour = Math.round(diffMin / 60);
  const diffDay = Math.round(diffHour / 24);
  
  if (diffDay > 0) {
    return `in ${diffDay} ${diffDay === 1 ? 'day' : 'days'}`;
  } else if (diffDay < 0) {
    return `${Math.abs(diffDay)} ${Math.abs(diffDay) === 1 ? 'day' : 'days'} ago`;
  } else if (diffHour > 0) {
    return `in ${diffHour} ${diffHour === 1 ? 'hour' : 'hours'}`;
  } else if (diffHour < 0) {
    return `${Math.abs(diffHour)} ${Math.abs(diffHour) === 1 ? 'hour' : 'hours'} ago`;
  } else if (diffMin > 0) {
    return `in ${diffMin} ${diffMin === 1 ? 'minute' : 'minutes'}`;
  } else if (diffMin < 0) {
    return `${Math.abs(diffMin)} ${Math.abs(diffMin) === 1 ? 'minute' : 'minutes'} ago`;
  } else {
    return 'just now';
  }
};

/**
 * Calculate duration between two dates in minutes
 * @param {Date|string} startDate - Start date object or ISO string
 * @param {Date|string} endDate - End date object or ISO string
 * @returns {number} Duration in minutes
 */
export const calculateDurationMinutes = (startDate, endDate) => {
  if (!startDate || !endDate) return 0;
  
  const start = typeof startDate === 'string' ? new Date(startDate) : startDate;
  const end = typeof endDate === 'string' ? new Date(endDate) : endDate;
  
  return Math.round((end - start) / (1000 * 60));
};
