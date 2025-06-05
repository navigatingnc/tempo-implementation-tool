from pydantic import BaseModel, Field, UUID4
from typing import Optional, List
from datetime import datetime
from enum import Enum

class TaskStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    cancelled = "cancelled"

class TaskBase(BaseModel):
    title: str
    description: Optional[str] = None
    status: TaskStatus = TaskStatus.pending
    priority: int = Field(3, ge=1, le=5)
    estimated_duration: Optional[int] = None  # in minutes
    due_date: Optional[datetime] = None
    tags: List[str] = []
    context: Optional[str] = None
    energy_required: Optional[int] = Field(None, ge=1, le=5)

class TaskCreate(TaskBase):
    pass

class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[TaskStatus] = None
    priority: Optional[int] = Field(None, ge=1, le=5)
    estimated_duration: Optional[int] = None
    due_date: Optional[datetime] = None
    tags: Optional[List[str]] = None
    context: Optional[str] = None
    energy_required: Optional[int] = Field(None, ge=1, le=5)

class Task(TaskBase):
    id: UUID4
    user_id: UUID4
    actual_duration: Optional[int] = None  # in minutes
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    recurring: bool = False
    recurrence_pattern: Optional[dict] = None

    class Config:
        orm_mode = True
