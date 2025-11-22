"""API Version 1 - Stable Release"""
from fastapi import APIRouter
from app.api.v1.endpoints import predictions, users, analytics, reports

router = APIRouter()
router.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
router.include_router(users.router, prefix="/users", tags=["users"])
router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
router.include_router(reports.router, prefix="/reports", tags=["reports"])
