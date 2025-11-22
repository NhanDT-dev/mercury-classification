"""V1 Report generation endpoints"""
from fastapi import APIRouter
from app.services.analytics.report_generator import ReportGenerator

router = APIRouter()

# Report generation and export
