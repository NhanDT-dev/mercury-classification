"""Health check and status endpoints"""
from fastapi import APIRouter
from datetime import datetime
from app.models.schemas import HealthCheck
from app.models.classifier import classifier
from app.core.config import settings

router = APIRouter(tags=["Health"])


@router.get("/", summary="Root Endpoint")
async def root():
    """Root endpoint - API welcome message"""
    return {
        "message": "Medical Text Classification API",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@router.get("/health", response_model=HealthCheck, summary="Health Check")
async def health_check():
    """
    Check API health status and model availability.

    No authentication required.

    **Returns:**
    - Service status
    - Model loading status
    - API version
    - Current timestamp
    """
    return HealthCheck(
        status="healthy" if classifier.is_loaded else "degraded",
        timestamp=datetime.utcnow(),
        model_loaded=classifier.is_loaded,
        model_name=classifier.model_name,
        version=settings.VERSION
    )


@router.get("/status", summary="Detailed Status")
async def status():
    """
    Get detailed service status including model information.

    No authentication required.
    """
    model_info = classifier.get_model_info()

    return {
        "service": "Medical Text Classification API",
        "version": settings.VERSION,
        "status": "operational" if classifier.is_loaded else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "model": {
            "loaded": model_info["is_loaded"],
            "name": model_info["name"],
            "type": model_info["type"],
            "device": model_info["device"]
        },
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "predictions": f"{settings.API_V1_PREFIX}/predict",
            "auth": f"{settings.API_V1_PREFIX}/auth"
        }
    }
