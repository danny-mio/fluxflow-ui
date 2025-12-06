#!/bin/bash
# FluxFlow UI - Setup Script
# Sets up the web interface for training and generation

set -e

echo "=== FluxFlow UI Setup ==="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python $required_version or higher is required (found $python_version)"
    exit 1
fi

echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, installing CUDA version..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "No GPU detected, installing CPU version..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install package
if [ "$1" == "--dev" ]; then
    echo "Installing in development mode with dev dependencies..."
    pip install -e .[dev]
else
    echo "Installing package..."
    pip install -e .
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To launch the UI:"
echo "  fluxflow-ui"
echo "  Or: ./launch.sh"
echo ""
echo "Then open http://localhost:5000 in your browser"
echo ""
