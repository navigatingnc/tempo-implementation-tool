from pydantic import BaseModel, Field, UUID4
from typing import Optional, List
from datetime import datetime
from enum import Enum

class CalendarEventBase(BaseModel):
    title: str
    description: Optional[str] = None
    start_time: datetime
    end_time: datetime
    location: Optional[str] = None
    is_all_day: bool = False

class CalendarEventCreate(CalendarEventBase):
    pass

class CalendarEventUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    location: Optional[str] = None
    is_all_day: Optional[bool] = None

class CalendarEvent(CalendarEventBase):
    id: UUID4
    user_id: UUID4
    created_at: datetime
    updated_at: datetime
    calendar_source: Optional[str] = None  # e.g., 'google', 'outlook', 'internal'
    external_id: Optional[str] = None  # ID from external calendar system
    attendees: Optional[List[dict]] = None
    recurring: bool = False
    recurrence_pattern: Optional[dict] = None
    reminder_minutes: Optional[List[int]] = None

    class Config:
        orm_mode = True
