#!/bin/bash
# FluxFlow UI Launcher
# Automatically uses correct venv and UI framework

set -e

# Load configuration if exists
if [ -f ".fluxflow_config" ]; then
    source .fluxflow_config
else
    echo "⚠️  Configuration not found. Running setup..."
    ./setup.sh
    source .fluxflow_config
fi

echo "==================================="
echo "FluxFlow UI Launcher"
echo "==================================="
echo "Python: $PYTHON_VERSION"
echo "UI Framework: $UI_FRAMEWORK"
echo "==================================="
echo ""

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: ./setup.sh"
    exit 1
fi

# Activate venv
echo "🔌 Activating virtual environment..."
source $VENV_PATH/bin/activate

# Verify dependencies
if ! python -c "import flask, torch, einops" 2>/dev/null; then
    echo "❌ Dependencies missing. Please run: ./setup.sh"
    exit 1
fi

echo ""
echo "🚀 Starting FluxFlow UI..."
echo "   Framework: $UI_FRAMEWORK"
echo "   URL: http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Launch appropriate UI
python $UI_APP
