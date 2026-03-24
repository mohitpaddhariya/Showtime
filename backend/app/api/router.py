"""Main API router — aggregates all versioned routes."""

from fastapi import APIRouter

from app.api.v1 import upload, process, status, download, events, jobs, preview

api_router = APIRouter()

api_router.include_router(upload.router, prefix="/v1", tags=["upload"])
api_router.include_router(process.router, prefix="/v1", tags=["process"])
api_router.include_router(status.router, prefix="/v1", tags=["status"])
api_router.include_router(download.router, prefix="/v1", tags=["download"])
api_router.include_router(events.router, prefix="/v1", tags=["events"])
api_router.include_router(jobs.router, prefix="/v1", tags=["jobs"])
api_router.include_router(preview.router, prefix="/v1", tags=["preview"])
