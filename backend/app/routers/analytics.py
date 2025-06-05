from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any
from sqlalchemy.orm import Session

from app.services.analytics_service import AnalyticsService
from app.services.auth import get_current_user
from app.db.session import get_db

router = APIRouter()

@router.get("/productivity")
def get_productivity_metrics(
    time_period: str = "week",
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get productivity metrics for the current user."""
    analytics_service = AnalyticsService(db)
    return analytics_service.get_productivity_metrics(current_user.id, time_period)

@router.get("/time-allocation")
def get_time_allocation(
    time_period: str = "week",
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get time allocation breakdown for the current user."""
    analytics_service = AnalyticsService(db)
    return analytics_service.get_time_allocation(current_user.id, time_period)

@router.get("/task-completion")
def get_task_completion_stats(
    time_period: str = "week",
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get task completion statistics for the current user."""
    analytics_service = AnalyticsService(db)
    return analytics_service.get_task_completion_stats(current_user.id, time_period)

@router.get("/focus-sessions")
def get_focus_session_stats(
    time_period: str = "week",
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get focus session statistics for the current user."""
    analytics_service = AnalyticsService(db)
    return analytics_service.get_focus_session_stats(current_user.id, time_period)

@router.get("/dashboard")
def get_dashboard_data(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get aggregated dashboard data for the current user."""
    analytics_service = AnalyticsService(db)
    return analytics_service.get_dashboard_data(current_user.id)
