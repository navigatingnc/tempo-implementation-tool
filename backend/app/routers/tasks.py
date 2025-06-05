from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session

from app.models.task import Task, TaskCreate, TaskUpdate
from app.services.task_service import TaskService
from app.services.auth import get_current_user
from app.db.session import get_db

router = APIRouter()

@router.post("/", response_model=Task)
def create_task(
    task: TaskCreate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new task."""
    task_service = TaskService(db)
    return task_service.create_task(current_user.id, task)

@router.get("/", response_model=List[Task])
def get_tasks(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all tasks for the current user."""
    task_service = TaskService(db)
    return task_service.get_tasks(current_user.id, skip, limit, status)

@router.get("/{task_id}", response_model=Task)
def get_task(
    task_id: UUID,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific task by ID."""
    task_service = TaskService(db)
    return task_service.get_task(current_user.id, task_id)

@router.put("/{task_id}", response_model=Task)
def update_task(
    task_id: UUID,
    task: TaskUpdate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a task."""
    task_service = TaskService(db)
    return task_service.update_task(current_user.id, task_id, task)

@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(
    task_id: UUID,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a task."""
    task_service = TaskService(db)
    task_service.delete_task(current_user.id, task_id)
    return None

@router.post("/{task_id}/complete", response_model=Task)
def complete_task(
    task_id: UUID,
    actual_duration: Optional[int] = None,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark a task as completed."""
    task_service = TaskService(db)
    return task_service.complete_task(current_user.id, task_id, actual_duration)
