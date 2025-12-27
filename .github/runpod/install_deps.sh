#!/bin/bash
set -e

echo "=== Installing Dependencies on RunPod ==="

pip install --upgrade pip --quiet

echo "Installing TotalSegmentator and ML libraries..."
pip install --no-cache-dir \
    totalsegmentator>=2.0.0 \
    nibabel>=5.0.0 \
    SimpleITK>=2.2.0 \
    pydicom>=2.3.0 \
    opencv-python-headless>=4.7.0 \
    scikit-image>=0.19.0 \
    scipy>=1.9.0 \
    tqdm>=4.65.0 \
    matplotlib>=3.7.0 \
    monai>=1.3.0

echo ""
echo "=== Installation Complete ==="
python -c "import totalsegmentator; print(f'TotalSegmentator: {totalsegmentator.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Creating directories..."
mkdir -p /workspace/amos22 /workspace/teacher_labels
echo "Ready to receive data!"
