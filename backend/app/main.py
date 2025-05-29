from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os

from app.routers import tasks, calendar, analytics, integrations
from app.models.user import User
from app.services.auth import get_current_user

app = FastAPI(
    title="Tempo API",
    description="Productivity and Time Management Agent API",
    version="0.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])
app.include_router(calendar.router, prefix="/api/calendar", tags=["calendar"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(integrations.router, prefix="/api/integrations", tags=["integrations"])

@app.get("/")
async def root():
    return {"message": "Welcome to Tempo API - Your Productivity and Time Management Agent"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}

@app.get("/api/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
