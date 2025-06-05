from sqlalchemy.orm import Session
from uuid import UUID
from datetime import datetime
from typing import List, Optional

from app.models.task import Task, TaskCreate, TaskUpdate, TaskStatus

class TaskService:
    def __init__(self, db: Session):
        self.db = db
        
    def create_task(self, user_id: UUID, task: TaskCreate) -> Task:
        """Create a new task for a user."""
        db_task = Task(
            **task.dict(),
            user_id=user_id,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self.db.add(db_task)
        self.db.commit()
        self.db.refresh(db_task)
        return db_task
        
    def get_tasks(self, user_id: UUID, skip: int = 0, limit: int = 100, status: Optional[str] = None) -> List[Task]:
        """Get all tasks for a user, optionally filtered by status."""
        query = self.db.query(Task).filter(Task.user_id == user_id)
        
        if status:
            query = query.filter(Task.status == status)
            
        return query.order_by(Task.created_at.desc()).offset(skip).limit(limit).all()
        
    def get_task(self, user_id: UUID, task_id: UUID) -> Task:
        """Get a specific task by ID."""
        task = self.db.query(Task).filter(Task.id == task_id, Task.user_id == user_id).first()
        if not task:
            raise ValueError("Task not found")
        return task
        
    def update_task(self, user_id: UUID, task_id: UUID, task_update: TaskUpdate) -> Task:
        """Update a task."""
        task = self.get_task(user_id, task_id)
        
        update_data = task_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(task, key, value)
            
        task.updated_at = datetime.now()
        self.db.commit()
        self.db.refresh(task)
        return task
        
    def delete_task(self, user_id: UUID, task_id: UUID) -> None:
        """Delete a task."""
        task = self.get_task(user_id, task_id)
        self.db.delete(task)
        self.db.commit()
        
    def complete_task(self, user_id: UUID, task_id: UUID, actual_duration: Optional[int] = None) -> Task:
        """Mark a task as completed."""
        task = self.get_task(user_id, task_id)
        task.status = TaskStatus.completed
        task.completed_at = datetime.now()
        task.actual_duration = actual_duration
        task.updated_at = datetime.now()
        self.db.commit()
        self.db.refresh(task)
        return task
