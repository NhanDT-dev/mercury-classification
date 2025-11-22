"""V1 Prediction endpoints - Production stable"""
from fastapi import APIRouter, Depends
from app.services.ml.inference_service import InferenceService
from app.schemas.requests.prediction_request import PredictionRequest
from app.core.security import verify_api_key

router = APIRouter()

# Advanced prediction endpoints for V1
