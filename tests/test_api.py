"""Test cases for API endpoints"""
import pytest
from httpx import AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test root endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Medical Text Classification API" in data["message"]


@pytest.mark.asyncio
async def test_health_check():
    """Test health check endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data


@pytest.mark.asyncio
async def test_status_endpoint():
    """Test status endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "model" in data


@pytest.mark.asyncio
async def test_docs_endpoint():
    """Test API documentation endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/docs")
        assert response.status_code == 200
