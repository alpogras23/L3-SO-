#!/bin/bash
# RunPod Teacher Generation - Quick Start Commands

echo "=== RunPod Teacher Generation Setup ==="
echo ""
echo "Pod ID: f3a35xebv5kqve"
echo "GPU: RTX 3090 (24GB)"
echo "Cost: $0.22/hour"
echo ""

# Kurulum sonrası doğrulama komutları:
echo "1. Kurulumu Doğrula:"
echo "   python -c 'import totalsegmentator; print(\"TotalSegmentator:\", totalsegmentator.__version__)'"
echo "   python -c 'import torch; print(\"PyTorch:\", torch.__version__, \"CUDA:\", torch.cuda.is_available())'"
echo "   python -c 'import torch; print(\"GPU:\", torch.cuda.get_device_name(0))'"
echo ""

# Dizin oluşturma
echo "2. Dizinleri Oluştur:"
echo "   mkdir -p /workspace/amos22 /workspace/teacher_labels /workspace/scripts"
echo ""

# Dosya upload (local'den çalıştırılacak)
echo "3. Local Makineden Dosya Upload:"
echo "   # Python script"
echo "   cat .github/azure-ml/step1_real_teachers_all.py | ssh f3a35xebv5kqve-64411d43@ssh.runpod.io -i ~/.ssh/id_ed25519_runpod 'cat > /workspace/step1_real_teachers_all.py'"
echo ""
echo "   # Core mini"
echo "   cat core_mini.py | ssh f3a35xebv5kqve-64411d43@ssh.runpod.io -i ~/.ssh/id_ed25519_runpod 'cat > /workspace/core_mini.py'"
echo ""
echo "   # AMOS22 data (BÜYÜK - rsync önerilir)"
echo "   rsync -avz --progress ~/Desktop/amos22/imagesTr/*.nii.gz f3a35xebv5kqve-64411d43@ssh.runpod.io:/workspace/amos22/"
echo ""

# Teacher generation başlatma
echo "4. Teacher Generation Başlat:"
echo "   cd /workspace"
echo "   python step1_real_teachers_all.py --input-dir amos22 --output-dir teacher_labels --num-workers 4"
echo ""

# İlerleme takibi
echo "5. İlerlemeyi İzle:"
echo "   watch -n 30 'ls teacher_labels/*.nii.gz | wc -l'"
echo "   nvidia-smi"
echo "   tail -f /workspace/teacher_generation.log"
echo ""

# Sonuçları indirme (local'den)
echo "6. Sonuçları İndir (Local Makineden):"
echo "   rsync -avz --progress f3a35xebv5kqve-64411d43@ssh.runpod.io:/workspace/teacher_labels/ ~/Desktop/L3_SO_ANALYSIS/amos22_teachers/"
echo ""

echo "=== Hızlı Test (1 vaka) ==="
echo "# Tek dosya ile test et:"
echo "python step1_real_teachers_all.py --input-dir amos22 --output-dir test_output --limit 1"
echo ""
