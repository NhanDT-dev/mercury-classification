"""Prediction endpoints for medical text classification"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict
import time
from app.models.schemas import (
    TextInput,
    BatchTextInput,
    PredictionResult,
    BatchPredictionResult,
    ModelInfo
)
from app.models.classifier import classifier
from app.core.security import verify_api_key, verify_bearer_token
from app.utils.logger import logger
from app.utils.exceptions import PredictionError

router = APIRouter(prefix="/predict", tags=["Predictions"])


@router.post(
    "/single",
    response_model=PredictionResult,
    summary="Classify Single Medical Text",
    description="Classify sentiment of a single medical text using AI model"
)
async def predict_single(
    input_data: TextInput,
    api_key: Dict = Depends(verify_api_key)
):
    """
    Classify sentiment of a single medical text.

    **Authentication:** Requires valid API Key in `X-API-Key` header.

    **Example API Keys:**
    - `demo-api-key-12345` (premium tier)
    - `test-api-key-67890` (basic tier)

    **Returns:**
    - Sentiment label (POSITIVE/NEGATIVE)
    - Confidence score (0-1)
    - Scores for all labels
    - Processing time in milliseconds
    """
    try:
        logger.info(f"Single prediction request from user: {api_key['user']}")

        # Run prediction
        label, confidence, scores, proc_time = classifier.predict(input_data.text)

        result = PredictionResult(
            text=input_data.text[:100] + "..." if len(input_data.text) > 100 else input_data.text,
            label=label,
            confidence=confidence,
            scores=scores,
            processing_time_ms=proc_time,
            model_version=classifier.get_model_info()["version"]
        )

        logger.info(f"Prediction completed: {label} ({confidence:.4f})")
        return result

    except PredictionError as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during prediction"
        )


@router.post(
    "/batch",
    response_model=BatchPredictionResult,
    summary="Classify Multiple Medical Texts",
    description="Classify sentiment of multiple medical texts in a single request"
)
async def predict_batch(
    input_data: BatchTextInput,
    api_key: Dict = Depends(verify_api_key)
):
    """
    Classify sentiment of multiple medical texts.

    **Authentication:** Requires valid API Key in `X-API-Key` header.

    **Limits:**
    - Maximum 50 texts per request
    - Each text max 5000 characters

    **Returns:**
    - List of prediction results
    - Total processing time
    - Number of texts processed
    """
    try:
        logger.info(f"Batch prediction request from user: {api_key['user']} ({len(input_data.texts)} texts)")

        start_time = time.time()

        # Run batch prediction
        results = classifier.predict_batch(input_data.texts)

        # Build response
        predictions = []
        for text, (label, confidence, scores, proc_time) in zip(input_data.texts, results):
            predictions.append(
                PredictionResult(
                    text=text[:100] + "..." if len(text) > 100 else text,
                    label=label,
                    confidence=confidence,
                    scores=scores,
                    processing_time_ms=proc_time,
                    model_version=classifier.get_model_info()["version"]
                )
            )

        total_time = (time.time() - start_time) * 1000

        logger.info(f"Batch prediction completed: {len(predictions)} texts in {total_time:.2f}ms")

        return BatchPredictionResult(
            predictions=predictions,
            total_processed=len(predictions),
            total_processing_time_ms=total_time,
            model_version=classifier.get_model_info()["version"]
        )

    except PredictionError as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during batch prediction"
        )


@router.get(
    "/model-info",
    response_model=ModelInfo,
    summary="Get Model Information",
    description="Get information about the loaded AI model"
)
async def get_model_info(api_key: Dict = Depends(verify_api_key)):
    """
    Get information about the loaded AI model.

    **Authentication:** Requires valid API Key in `X-API-Key` header.

    **Returns:**
    - Model name and version
    - Supported labels
    - Maximum input length
    - Model description
    """
    try:
        info = classifier.get_model_info()

        return ModelInfo(
            name=info["name"],
            version=info["version"],
            type=info["type"],
            max_length=info["max_length"],
            labels=info["labels"],
            description=info["description"]
        )

    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information"
        )
