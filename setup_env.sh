#!/bin/bash
# Setup script for ConfSteer conda environment
# Usage: bash setup_env.sh

set -e

ENV_NAME="confsteer"
PYTHON_VERSION="3.10"

echo "=========================================="
echo "  ConfSteer Environment Setup"
echo "  Env: ${ENV_NAME}, Python: ${PYTHON_VERSION}"
echo "=========================================="

# Create conda environment
echo ""
echo "[1] Creating conda environment..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

# Activate
echo ""
echo "[2] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Core dependencies
echo ""
echo "[3] Installing dependencies..."
pip install --upgrade pip

# ML / data
pip install numpy scipy pandas scikit-learn

# HDF5 for hidden states
pip install h5py

# Deep learning (for future MLP extension)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Utilities
pip install tqdm matplotlib

echo ""
echo "=========================================="
echo "  Done! Activate with:"
echo "    conda activate ${ENV_NAME}"
echo "=========================================="
