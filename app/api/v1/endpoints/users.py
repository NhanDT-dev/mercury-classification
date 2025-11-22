"""V1 User management endpoints"""
from fastapi import APIRouter, Depends
from app.db.models.user import User
from app.services.user_service import UserService

router = APIRouter()

# User CRUD operations
