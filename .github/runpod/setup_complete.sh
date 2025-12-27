#!/bin/bash
# RunPod Complete Setup Script
# Tek komutla çalıştırılabilir

set -e

echo "=== RunPod Teacher Generation Setup ==="
echo ""

# 1. Verify installation
echo "1. Verifying Python packages..."
python -c "import totalsegmentator; print('✓ TotalSegmentator:', totalsegmentator.__version__)"
python -c "import torch; print('✓ PyTorch:', torch.__version__)"
python -c "import torch; print('✓ CUDA Available:', torch.cuda.is_available())"
python -c "import torch; print('✓ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ""

# 2. Create directories
echo "2. Creating directories..."
mkdir -p /workspace/amos22 /workspace/teacher_labels
echo "✓ Directories created"

# 3. Check disk space
echo ""
echo "3. Disk space:"
df -h | grep -E '(Filesystem|/$|/workspace)' || df -h | head -5

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Upload AMOS22 files to /workspace/amos22/"
echo "2. Upload step1_real_teachers_all.py to /workspace/"
echo "3. Run: python /workspace/step1_real_teachers_all.py --input-dir /workspace/amos22 --output-dir /workspace/teacher_labels"
echo ""
echo "Current status:"
ls -lh /workspace/ 2>/dev/null || echo "Workspace directory listing unavailable"
