"""SQLAlchemy Base class and database setup"""
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

Base = declarative_base()

# Import all models here for Alembic migrations
from app.db.models.user import User
from app.db.models.prediction import Prediction
from app.db.models.api_key import APIKey
from app.db.models.analytics import Analytics
from app.db.models.audit_log import AuditLog
