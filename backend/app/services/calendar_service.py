from sqlalchemy.orm import Session
from uuid import UUID
from datetime import datetime
from typing import List, Optional

from app.models.calendar_event import CalendarEvent, CalendarEventCreate, CalendarEventUpdate

class CalendarService:
    def __init__(self, db: Session):
        self.db = db
        
    def create_event(self, user_id: UUID, event: CalendarEventCreate) -> CalendarEvent:
        """Create a new calendar event for a user."""
        db_event = CalendarEvent(
            **event.dict(),
            user_id=user_id,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self.db.add(db_event)
        self.db.commit()
        self.db.refresh(db_event)
        return db_event
        
    def get_events(self, user_id: UUID, skip: int = 0, limit: int = 100, 
                  start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[CalendarEvent]:
        """Get all calendar events for a user, optionally filtered by date range."""
        query = self.db.query(CalendarEvent).filter(CalendarEvent.user_id == user_id)
        
        if start_date:
            start = datetime.fromisoformat(start_date)
            query = query.filter(CalendarEvent.start_time >= start)
            
        if end_date:
            end = datetime.fromisoformat(end_date)
            query = query.filter(CalendarEvent.end_time <= end)
            
        return query.order_by(CalendarEvent.start_time).offset(skip).limit(limit).all()
        
    def get_event(self, user_id: UUID, event_id: UUID) -> CalendarEvent:
        """Get a specific calendar event by ID."""
        event = self.db.query(CalendarEvent).filter(
            CalendarEvent.id == event_id, 
            CalendarEvent.user_id == user_id
        ).first()
        if not event:
            raise ValueError("Calendar event not found")
        return event
        
    def update_event(self, user_id: UUID, event_id: UUID, event_update: CalendarEventUpdate) -> CalendarEvent:
        """Update a calendar event."""
        event = self.get_event(user_id, event_id)
        
        update_data = event_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(event, key, value)
            
        event.updated_at = datetime.now()
        self.db.commit()
        self.db.refresh(event)
        return event
        
    def delete_event(self, user_id: UUID, event_id: UUID) -> None:
        """Delete a calendar event."""
        event = self.get_event(user_id, event_id)
        self.db.delete(event)
        self.db.commit()
        
    def get_events_by_source(self, user_id: UUID, source: str, 
                           start_time: Optional[datetime] = None, 
                           end_time: Optional[datetime] = None) -> List[CalendarEvent]:
        """Get events from a specific source (e.g., 'google', 'outlook')."""
        query = self.db.query(CalendarEvent).filter(
            CalendarEvent.user_id == user_id,
            CalendarEvent.calendar_source == source
        )
        
        if start_time:
            query = query.filter(CalendarEvent.start_time >= start_time)
            
        if end_time:
            query = query.filter(CalendarEvent.end_time <= end_time)
            
        return query.all()
        
    def create_batch_events(self, events: List[CalendarEventCreate]) -> List[CalendarEvent]:
        """Create multiple calendar events in a batch."""
        db_events = []
        for event in events:
            db_event = CalendarEvent(
                **event.dict(),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            self.db.add(db_event)
            db_events.append(db_event)
            
        self.db.commit()
        for event in db_events:
            self.db.refresh(event)
            
        return db_events
        
    def update_batch_events(self, events: List[dict]) -> List[CalendarEvent]:
        """Update multiple calendar events in a batch."""
        updated_events = []
        for event_data in events:
            event_id = event_data['id']
            update_data = event_data['data']
            
            event = self.db.query(CalendarEvent).filter(CalendarEvent.id == event_id).first()
            if event:
                for key, value in update_data.items():
                    setattr(event, key, value)
                event.updated_at = datetime.now()
                updated_events.append(event)
                
        self.db.commit()
        for event in updated_events:
            self.db.refresh(event)
            
        return updated_events
        
    def delete_batch_events(self, user_id: UUID, event_ids: List[str]) -> None:
        """Delete multiple calendar events in a batch."""
        self.db.query(CalendarEvent).filter(
            CalendarEvent.id.in_(event_ids),
            CalendarEvent.user_id == user_id
        ).delete(synchronize_session=False)
        
        self.db.commit()
