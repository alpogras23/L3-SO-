#!/bin/bash
# RunPod Setup Script - Install dependencies and prepare environment

set -e  # Exit on error

echo "=== RunPod L3 Teacher Generation Setup ==="
echo "Installing dependencies..."

# Update pip
pip install --upgrade pip

# Install core dependencies
echo "Installing TotalSegmentator and ML libraries..."
pip install --no-cache-dir \
    totalsegmentator>=2.0.0 \
    nibabel>=5.0.0 \
    SimpleITK>=2.2.0 \
    pydicom>=2.3.0 \
    opencv-python-headless>=4.7.0 \
    scikit-image>=0.19.0 \
    scipy>=1.9.0 \
    numpy>=1.21.0 \
    tqdm>=4.65.0 \
    matplotlib>=3.7.0 \
    monai>=1.3.0

# Verify GPU
echo "Checking GPU..."
nvidia-smi

# Create directories
echo "Creating directories..."
mkdir -p /workspace/amos22
mkdir -p /workspace/teacher_labels
mkdir -p /tmp/scratch

# Set environment variables
export PYTHONUNBUFFERED=1
export TOTALSEG_SCRATCH=/tmp/scratch
export SCRATCH=/tmp/scratch
export OMP_NUM_THREADS=8

echo "=== Setup Complete ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""
echo "Next steps:"
echo "1. Upload AMOS22 data to /workspace/amos22/"
echo "2. Upload step1_real_teachers_all.py to /workspace/"
echo "3. Run: python step1_real_teachers_all.py"
