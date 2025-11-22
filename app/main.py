"""Main FastAPI application"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import time

from app.core.config import settings
from app.routes import health, auth, predictions
from app.models.classifier import classifier
from app.utils.logger import logger
from app.utils.exceptions import (
    validation_exception_handler,
    generic_exception_handler,
    model_load_exception_handler,
    prediction_exception_handler,
    ModelLoadError,
    PredictionError
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager - handles startup and shutdown events.

    Startup:
    - Load ML model
    - Initialize resources

    Shutdown:
    - Cleanup resources
    - Unload model
    """
    # Startup
    logger.info("=" * 60)
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info("=" * 60)

    try:
        # Load the ML model
        logger.info("Loading AI model...")
        classifier.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.warning("API will start but predictions will fail until model is loaded")

    logger.info(f"API is ready at {settings.API_V1_PREFIX}")
    logger.info("Documentation available at /docs")

    yield

    # Shutdown
    logger.info("Shutting down application...")
    classifier.unload_model()
    logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


# Middleware
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response


# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code} - {request.method} {request.url.path}")
    return response


# Exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(ModelLoadError, model_load_exception_handler)
app.add_exception_handler(PredictionError, prediction_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)


# Include routers
app.include_router(health.router)
app.include_router(auth.router, prefix=settings.API_V1_PREFIX)
app.include_router(predictions.router, prefix=settings.API_V1_PREFIX)


# Root redirect
@app.get("/favicon.ico")
async def favicon():
    """Favicon endpoint"""
    return JSONResponse({"message": "No favicon"}, status_code=204)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
