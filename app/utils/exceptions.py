"""Custom exceptions and error handlers"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from typing import Union
import traceback
from app.utils.logger import logger


class ModelLoadError(Exception):
    """Raised when model fails to load"""
    pass


class PredictionError(Exception):
    """Raised when prediction fails"""
    pass


class AuthenticationError(Exception):
    """Raised when authentication fails"""
    pass


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "path": str(request.url.path)
        }
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """Handle generic exceptions"""
    logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again later.",
            "path": str(request.url.path)
        }
    )


async def model_load_exception_handler(request: Request, exc: ModelLoadError):
    """Handle model loading errors"""
    logger.error(f"Model load error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "Model Load Error",
            "detail": "AI model is currently unavailable. Please try again later.",
            "path": str(request.url.path)
        }
    )


async def prediction_exception_handler(request: Request, exc: PredictionError):
    """Handle prediction errors"""
    logger.error(f"Prediction error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Prediction Error",
            "detail": str(exc),
            "path": str(request.url.path)
        }
    )
