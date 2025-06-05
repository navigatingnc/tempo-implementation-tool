from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import List, Dict, Any
from sqlalchemy.orm import Session

from app.services.integration_service import IntegrationService
from app.services.auth import get_current_user
from app.db.session import get_db

router = APIRouter()

@router.get("/")
def list_integrations(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all active integrations for the current user."""
    integration_service = IntegrationService(db)
    return integration_service.get_user_integrations(current_user.id)

@router.get("/google/login")
async def google_login(
    request: Request,
    current_user = Depends(get_current_user)
):
    """Initiate Google Calendar integration via OAuth2."""
    # This would normally redirect to Google OAuth
    return {"message": "OAuth flow would start here", "redirect_url": "https://accounts.google.com/o/oauth2/auth"}

@router.get("/google/callback")
async def google_callback(
    request: Request,
    code: str,
    db: Session = Depends(get_db)
):
    """Handle the OAuth2 callback from Google."""
    # This would normally process the OAuth callback
    return {"message": "OAuth callback received", "code": code}

@router.delete("/{integration_id}")
def delete_integration(
    integration_id: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete an integration."""
    integration_service = IntegrationService(db)
    integration_service.delete_integration(current_user.id, integration_id)
    return {"message": "Integration deleted successfully"}

@router.post("/{integration_id}/sync")
def trigger_sync(
    integration_id: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Manually trigger synchronization for an integration."""
    integration_service = IntegrationService(db)
    integration_service.trigger_sync(current_user.id, integration_id)
    return {"message": "Sync triggered successfully"}
