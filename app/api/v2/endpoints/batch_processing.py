"""V2 Large-scale batch processing endpoints"""
from fastapi import APIRouter, BackgroundTasks
from app.workers.celery_tasks.batch_prediction_task import BatchPredictionTask

router = APIRouter()

# Asynchronous batch processing with Celery
