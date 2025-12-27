#!/bin/bash
# RunPod Web Terminal'de çalıştırılacak komutlar

echo "=== Kurulum Doğrulama ==="
python -c "import totalsegmentator; print('TotalSegmentator:', totalsegmentator.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

echo ""
echo "=== Dizinler Oluşturuluyor ==="
mkdir -p /workspace/amos22 /workspace/teacher_labels
ls -ld /workspace/amos22 /workspace/teacher_labels

echo ""
echo "=== Disk Durumu ==="
df -h | head -5

echo ""
echo "=== Python Script Download (GitHub'dan) ==="
cd /workspace
wget -O step1_real_teachers_all.py https://raw.githubusercontent.com/alpogras23/L3-SO-/psoas-improvement/.github/azure-ml/step1_real_teachers_all.py
chmod +x step1_real_teachers_all.py
ls -lh step1_real_teachers_all.py

echo ""
echo "=== Test: 1 vaka ile hızlı test ==="
echo "Bu komutu AMOS22 verileri yüklendikten sonra çalıştırın:"
echo "python step1_real_teachers_all.py --input-dir amos22 --output-dir test_output --limit 1"
