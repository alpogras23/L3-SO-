#!/bin/bash
# TS segmentasyon durumunu kontrol eder

echo "=== TotalSegmentator İlerleme Kontrolü ==="
echo ""

# Çalışan süreçleri say
running=$(ps aux | grep "TotalSegmentator" | grep -v grep | wc -l)
echo "Çalışan TS süreçleri: $running"
echo ""

# Her vaka için çıktıları kontrol et
for case in amos_0001 amos_0004 amos_0005; do
    echo "--- $case ---"
    out_dir="data/ts_labels_amos/$case"
    
    if [ ! -d "$out_dir" ]; then
        echo "  ❌ Klasör yok"
    else
        file_count=$(ls "$out_dir"/*.nii.gz 2>/dev/null | wc -l)
        if [ $file_count -eq 0 ]; then
            echo "  ⏳ Henüz tamamlanmadı (0 dosya)"
        else
            echo "  ✅ $file_count dosya üretildi:"
            ls "$out_dir"/*.nii.gz 2>/dev/null | xargs -n1 basename
        fi
    fi
    echo ""
done

echo "=== Beklenen Dosyalar ==="
echo "  - vertebrae_L3.nii.gz"
echo "  - psoas_major_left.nii.gz"
echo "  - psoas_major_right.nii.gz"
echo "  - subcutaneous_fat.nii.gz"
echo "  - torso_fat.nii.gz"
