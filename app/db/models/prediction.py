"""Prediction history database model"""
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.base import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    input_text = Column(Text, nullable=False)
    predicted_label = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    all_scores = Column(JSON, nullable=False)
    model_version = Column(String, nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    api_endpoint = Column(String)
    metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship
    user = relationship("User", backref="predictions")
