from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session

from app.models.calendar_event import CalendarEvent, CalendarEventCreate, CalendarEventUpdate
from app.services.calendar_service import CalendarService
from app.services.auth import get_current_user
from app.db.session import get_db

router = APIRouter()

@router.post("/", response_model=CalendarEvent)
def create_calendar_event(
    event: CalendarEventCreate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new calendar event."""
    calendar_service = CalendarService(db)
    return calendar_service.create_event(current_user.id, event)

@router.get("/", response_model=List[CalendarEvent])
def get_calendar_events(
    skip: int = 0,
    limit: int = 100,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all calendar events for the current user."""
    calendar_service = CalendarService(db)
    return calendar_service.get_events(current_user.id, skip, limit, start_date, end_date)

@router.get("/{event_id}", response_model=CalendarEvent)
def get_calendar_event(
    event_id: UUID,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific calendar event by ID."""
    calendar_service = CalendarService(db)
    return calendar_service.get_event(current_user.id, event_id)

@router.put("/{event_id}", response_model=CalendarEvent)
def update_calendar_event(
    event_id: UUID,
    event: CalendarEventUpdate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a calendar event."""
    calendar_service = CalendarService(db)
    return calendar_service.update_event(current_user.id, event_id, event)

@router.delete("/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_calendar_event(
    event_id: UUID,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a calendar event."""
    calendar_service = CalendarService(db)
    calendar_service.delete_event(current_user.id, event_id)
    return None
