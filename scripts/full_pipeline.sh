#!/bin/bash
# Full Training Pipeline
# TotalSegmentator tamamlandıktan sonra çalıştırılacak

set -e

cd ~/L3_SO_ANALYSIS
source .venv/bin/activate

echo "=== Pipeline Başladı ==="
date

# 1. TotalSegmentator durumunu kontrol et
TOTAL_CT=$(ls ~/amos22_data/imagesTr/*.nii.gz | wc -l)
COMPLETED=$(ls -d ~/amos22_data/totalseg_output/amos_* 2>/dev/null | wc -l)

echo "TotalSegmentator: $COMPLETED / $TOTAL_CT tamamlandı"

if [ "$COMPLETED" -lt 100 ]; then
    echo "En az 100 vaka bekleniyor, mevcut: $COMPLETED"
    echo "TotalSegmentator tamamlanmasını bekleyin."
    exit 1
fi

# 2. L3 Slice Çıkarımı
echo ""
echo "=== L3 Slice Çıkarımı ==="
python scripts/extract_l3_slices.py

# 3. Eğitim Dataset Kontrolü
SLICE_COUNT=$(ls ~/amos22_data/l3_slices_2d/ct_slices/*.npy 2>/dev/null | wc -l)
echo "Toplam $SLICE_COUNT eğitim slice'ı hazır"

if [ "$SLICE_COUNT" -lt 50 ]; then
    echo "En az 50 slice gerekli, mevcut: $SLICE_COUNT"
    exit 1
fi

# 4. U-Net Eğitimi
echo ""
echo "=== U-Net Eğitimi Başlıyor ==="
python scripts/train_unet_vfa_pma.py

echo ""
echo "=== Pipeline Tamamlandı ==="
date
echo "Model: ~/L3_SO_ANALYSIS/models/psoas_vfa_best.pth"
