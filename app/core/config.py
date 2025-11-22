"""Application configuration and settings"""
from pydantic_settings import BaseSettings
from typing import List
import secrets


class Settings(BaseSettings):
    """Application settings and configuration"""

    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Medical Text Classification API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "AI-powered medical text classification service with sentiment analysis"

    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    API_KEY_NAME: str = "X-API-Key"

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Model Configuration
    MODEL_NAME: str = "distilbert-base-uncased-finetuned-sst-2-english"
    MODEL_CACHE_DIR: str = "./model_cache"
    MAX_TEXT_LENGTH: int = 512

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
