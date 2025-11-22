"""V2 Real-time streaming predictions via WebSocket"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.ml.streaming_service import StreamingPredictionService

router = APIRouter()

# WebSocket endpoints for real-time predictions
