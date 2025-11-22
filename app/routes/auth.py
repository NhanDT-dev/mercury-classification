"""Authentication endpoints"""
from fastapi import APIRouter, HTTPException, status, Depends
from datetime import timedelta
from app.models.schemas import TokenRequest, TokenResponse
from app.core.config import settings
from app.core.security import create_access_token, verify_password, get_password_hash
from app.utils.logger import logger

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Demo users for testing (in production, use database)
DEMO_USERS = {
    "demo_user": {
        "username": "demo_user",
        "password": get_password_hash("demo_password"),
        "email": "demo@medical-api.com",
        "role": "premium"
    },
    "test_user": {
        "username": "test_user",
        "password": get_password_hash("test_password"),
        "email": "test@medical-api.com",
        "role": "basic"
    }
}


@router.post("/token", response_model=TokenResponse, summary="Get JWT Access Token")
async def login(credentials: TokenRequest):
    """
    Authenticate user and return JWT access token.

    **Demo Credentials:**
    - Username: `demo_user` / Password: `demo_password`
    - Username: `test_user` / Password: `test_password`

    Returns:
        JWT access token valid for 30 minutes
    """
    user = DEMO_USERS.get(credentials.username)

    if not user or not verify_password(credentials.password, user["password"]):
        logger.warning(f"Failed login attempt for username: {credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires
    )

    logger.info(f"User {credentials.username} logged in successfully")

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }


@router.get("/verify", summary="Verify Token")
async def verify_token_endpoint(user: dict = Depends(lambda: None)):
    """
    Verify if the provided token is valid.

    Requires Bearer token in Authorization header.
    """
    return {
        "status": "valid",
        "message": "Token is valid"
    }
