@echo off
echo ============================================================
echo Medical Text Classification API - Setup Script
echo ============================================================
echo.

echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo Done.
echo.

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
echo Done.
echo.

echo [3/4] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Done.
echo.

echo [4/4] Creating necessary directories...
if not exist logs mkdir logs
if not exist model_cache mkdir model_cache
echo Done.
echo.

echo ============================================================
echo Setup complete!
echo ============================================================
echo.
echo To run the application:
echo   1. Activate virtual environment: venv\Scripts\activate
echo   2. Run server: python run.py
echo   3. Open browser: http://localhost:8000/docs
echo.
echo For testing:
echo   - Run tests: pytest
echo   - Import Postman collection: postman_collection.json
echo.
pause
