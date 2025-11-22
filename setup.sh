#!/bin/bash

echo "============================================================"
echo "Medical Text Classification API - Setup Script"
echo "============================================================"
echo ""

echo "[1/4] Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi
echo "Done."
echo ""

echo "[2/4] Activating virtual environment..."
source venv/bin/activate
echo "Done."
echo ""

echo "[3/4] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi
echo "Done."
echo ""

echo "[4/4] Creating necessary directories..."
mkdir -p logs model_cache
echo "Done."
echo ""

echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "To run the application:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run server: python run.py"
echo "  3. Open browser: http://localhost:8000/docs"
echo ""
echo "For testing:"
echo "  - Run tests: pytest"
echo "  - Import Postman collection: postman_collection.json"
echo ""
