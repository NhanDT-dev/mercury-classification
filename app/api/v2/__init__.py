"""API Version 2 - Beta Features"""
from fastapi import APIRouter
from app.api.v2.endpoints import predictions_v2, streaming, batch_processing

router = APIRouter()
router.include_router(predictions_v2.router, prefix="/predictions", tags=["predictions-v2"])
router.include_router(streaming.router, prefix="/streaming", tags=["streaming"])
router.include_router(batch_processing.router, prefix="/batch", tags=["batch"])
