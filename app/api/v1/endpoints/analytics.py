"""V1 Analytics and metrics endpoints"""
from fastapi import APIRouter
from app.services.analytics.metrics_service import MetricsService
from app.services.analytics.reporting_service import ReportingService

router = APIRouter()

# Analytics and business intelligence endpoints
