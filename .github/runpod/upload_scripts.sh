#!/bin/bash
# Upload files to RunPod via SSH pipe (since SCP doesn't work)

POD_SSH="f3a35xebv5kqve-64411d43@ssh.runpod.io"
SSH_KEY="~/.ssh/id_ed25519_runpod"

echo "=== Uploading Files to RunPod ==="

# 1. Upload Python script
echo "1. Uploading step1_real_teachers_all.py..."
cat .github/azure-ml/step1_real_teachers_all.py | \
    ssh -i $SSH_KEY $POD_SSH 'cat > /workspace/step1_real_teachers_all.py && echo "✓ Python script uploaded"'

# 2. Upload core_mini.py
echo "2. Uploading core_mini.py..."
cat core_mini.py | \
    ssh -i $SSH_KEY $POD_SSH 'cat > /workspace/core_mini.py && echo "✓ core_mini uploaded"'

# 3. Upload HU calibration modules (if needed)
if [ -f hu_calibration.py ]; then
    echo "3. Uploading hu_calibration.py..."
    cat hu_calibration.py | \
        ssh -i $SSH_KEY $POD_SSH 'cat > /workspace/hu_calibration.py && echo "✓ HU calibration uploaded"'
fi

if [ -f leak_prevention.py ]; then
    echo "4. Uploading leak_prevention.py..."
    cat leak_prevention.py | \
        ssh -i $SSH_KEY $POD_SSH 'cat > /workspace/leak_prevention.py && echo "✓ Leak prevention uploaded"'
fi

echo ""
echo "=== Files uploaded! ==="
echo ""
echo "Next: Upload AMOS22 data (this will take longer):"
echo "Run on LOCAL machine:"
echo "  ./upload_amos_data.sh"
