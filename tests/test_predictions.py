"""Test cases for prediction endpoints"""
import pytest
from httpx import AsyncClient
from app.main import app


API_KEY = "demo-api-key-12345"


@pytest.mark.asyncio
async def test_single_prediction_valid():
    """Test single text prediction with valid input"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/predict/single",
            headers={"X-API-Key": API_KEY},
            json={"text": "The patient responded very well to treatment"}
        )
        # Should return 200 if model loaded, 503 if not
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "label" in data
            assert "confidence" in data
            assert "scores" in data
            assert "processing_time_ms" in data


@pytest.mark.asyncio
async def test_single_prediction_invalid_input():
    """Test single text prediction with empty text"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/predict/single",
            headers={"X-API-Key": API_KEY},
            json={"text": ""}
        )
        assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_batch_prediction_valid():
    """Test batch prediction with valid input"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/predict/batch",
            headers={"X-API-Key": API_KEY},
            json={
                "texts": [
                    "Great results from the treatment",
                    "Patient experienced side effects",
                    "Recovery is on track"
                ]
            }
        )
        # Should return 200 if model loaded, 503 if not
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_processed" in data
            assert len(data["predictions"]) == 3


@pytest.mark.asyncio
async def test_batch_prediction_empty_list():
    """Test batch prediction with empty list"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/predict/batch",
            headers={"X-API-Key": API_KEY},
            json={"texts": []}
        )
        assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_model_info():
    """Test model info endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/predict/model-info",
            headers={"X-API-Key": API_KEY}
        )
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "labels" in data
