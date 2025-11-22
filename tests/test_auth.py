"""Test cases for authentication"""
import pytest
from httpx import AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_login_success():
    """Test successful login"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/token",
            json={"username": "demo_user", "password": "demo_password"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_invalid_credentials():
    """Test login with invalid credentials"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/token",
            json={"username": "invalid", "password": "wrong"}
        )
        assert response.status_code == 401


@pytest.mark.asyncio
async def test_prediction_with_api_key():
    """Test prediction endpoint with valid API key"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/predict/single",
            headers={"X-API-Key": "demo-api-key-12345"},
            json={"text": "This is a great treatment"}
        )
        # Will fail if model not loaded, but should return proper status
        assert response.status_code in [200, 503]


@pytest.mark.asyncio
async def test_prediction_without_api_key():
    """Test prediction endpoint without API key"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/predict/single",
            json={"text": "This is a test"}
        )
        assert response.status_code == 401
