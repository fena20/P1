#!/bin/bash
# Quick Start Script for Energy Optimization Framework
# Target: Applied Energy (Q1 Journal)

echo "=========================================="
echo "Energy Optimization Framework - Quick Start"
echo "=========================================="
echo ""

# Check Python version
echo "[1/4] Checking Python version..."
python_version=$(python --version 2>&1)
echo "   $python_version"

if ! command -v python &> /dev/null; then
    echo "   ✗ Python not found. Please install Python 3.8+"
    exit 1
fi

# Install dependencies
echo ""
echo "[2/4] Installing dependencies..."
echo "   This may take 2-3 minutes..."
pip install -q -r requirements.txt

if [ $? -eq 0 ]; then
    echo "   ✓ All dependencies installed successfully"
else
    echo "   ✗ Error installing dependencies"
    exit 1
fi

# Verify installation
echo ""
echo "[3/4] Verifying installation..."
python -c "import numpy, pandas, sklearn, xgboost, lightgbm, optuna, shap, pymoo" 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ All imports successful"
else
    echo "   ✗ Some imports failed"
    exit 1
fi

# Run the framework
echo ""
echo "[4/4] Running optimization framework..."
echo "   Expected runtime: 10-15 minutes"
echo "   (Press Ctrl+C to stop)"
echo ""
python energy_optimization_framework.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Framework execution complete!"
    echo "=========================================="
    echo ""
    echo "Generated files:"
    ls -lh *.png *.csv 2>/dev/null | awk '{print "  -", $9, "(" $5 ")"}'
    echo ""
    echo "Next steps:"
    echo "  1. Review generated visualizations (*.png)"
    echo "  2. Analyze performance table (*.csv)"
    echo "  3. Read USAGE_GUIDE.md for interpretation"
    echo ""
else
    echo ""
    echo "✗ Framework execution failed"
    echo "Check error messages above"
    exit 1
fi
