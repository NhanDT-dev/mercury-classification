"""Analytics and metrics database model"""
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Date
from sqlalchemy.sql import func
from app.db.base import Base

class Analytics(Base):
    __tablename__ = "analytics"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True)
    endpoint = Column(String, index=True)
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    avg_response_time_ms = Column(Float)
    total_predictions = Column(Integer, default=0)
    unique_users = Column(Integer, default=0)
    metrics_data = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
