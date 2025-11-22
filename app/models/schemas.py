"""Pydantic schemas for request/response validation"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum


class SentimentLabel(str, Enum):
    """Sentiment classification labels"""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


class TextInput(BaseModel):
    """Input schema for text classification"""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Medical text to classify",
        example="The patient responded very well to the treatment and shows significant improvement."
    )

    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class BatchTextInput(BaseModel):
    """Input schema for batch text classification"""
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="List of medical texts to classify",
        example=[
            "The treatment was very effective.",
            "Patient experienced severe side effects."
        ]
    )

    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError("Texts list cannot be empty")
        cleaned = [text.strip() for text in v if text and text.strip()]
        if not cleaned:
            raise ValueError("All texts are empty or whitespace only")
        return cleaned


class PredictionResult(BaseModel):
    """Output schema for single prediction"""
    text: str = Field(..., description="Input text that was classified")
    label: str = Field(..., description="Predicted sentiment label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")
    scores: Dict[str, float] = Field(..., description="Scores for all labels")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="Model version used for prediction")


class BatchPredictionResult(BaseModel):
    """Output schema for batch predictions"""
    predictions: List[PredictionResult] = Field(..., description="List of prediction results")
    total_processed: int = Field(..., description="Total number of texts processed")
    total_processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    model_version: str = Field(..., description="Model version used for predictions")


class HealthCheck(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    model_name: str = Field(..., description="Name of the loaded model")
    version: str = Field(..., description="API version")


class TokenRequest(BaseModel):
    """JWT token request"""
    username: str = Field(..., min_length=3, example="demo_user")
    password: str = Field(..., min_length=6, example="demo_password")


class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    path: str = Field(..., description="Request path")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class ModelInfo(BaseModel):
    """Model information"""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    type: str = Field(..., description="Model type")
    max_length: int = Field(..., description="Maximum input length")
    labels: List[str] = Field(..., description="Available classification labels")
    description: str = Field(..., description="Model description")
