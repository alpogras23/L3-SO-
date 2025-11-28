#!/bin/bash
# ============================================================
# COLAB FULL PIPELINE: TEACHER ÜRETIMI + EĞİTİM + QA
# ============================================================
# Bu script Colab GPU ortamında tüm pipeline'ı tek komutta çalıştırır:
#   1. TotalSegmentator teacher üretimi
#   2. AMOS + TBCC eğitim scriptleri
#   3. QA overlay üretimi ve metrik analizi
#
# Kullanım:
#   !bash scripts/run_all_colab.sh

set -e  # Hata durumunda dur

echo "============================================================"
echo "L3 VFA/PMA FULL PIPELINE BAŞLATILIYOR"
echo "============================================================"

# Drive mount kontrolü
if [ ! -d "/content/drive" ]; then
    echo "⚠️ Drive mount edilmemiş. Lütfen önce Drive'ı mount edin."
    exit 1
fi

# Ortam değişkenleri
DRIVE_ROOT="/content/drive/MyDrive"
AMOS_ROOT="${DRIVE_ROOT}/AMOS22"
TS_ROOT="${DRIVE_ROOT}/TS_teachers_AMOS22"
C2C_ROOT="${DRIVE_ROOT}/C2C_teachers_AMOS22"
TBCC_ROOT="${DRIVE_ROOT}/TBCC_data"
RESULTS_ROOT="${DRIVE_ROOT}/L3_RESULTS"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${RESULTS_ROOT}/full_run_${TIMESTAMP}"

mkdir -p "${RUN_DIR}"

echo "📁 Çıktı dizini: ${RUN_DIR}"

# ============================================================
# 1. TOTALSEGMENTATOR TEACHER ÜRETİMİ (VARSA ATLA)
# ============================================================
if [ ! -d "${TS_ROOT}" ] || [ -z "$(ls -A ${TS_ROOT})" ]; then
    echo ""
    echo "🔬 TotalSegmentator teacher üretimi başlıyor..."
    echo "   (İlk çalışma model ağırlıklarını indirecek, ~2GB)"
    
    # AMOS22 için TS
    if [ -d "${AMOS_ROOT}/imagesTr" ]; then
        python tools/generate_ts_teachers.py \
            --amos-root "${AMOS_ROOT}" \
            --out-dir "${TS_ROOT}" \
            --tasks abdominal_muscles total \
            --fast
        echo "✅ AMOS TS teacher üretimi tamamlandı"
    else
        echo "⚠️ AMOS22/imagesTr bulunamadı, TS teacher atlandı"
    fi
else
    echo "✅ TS teacher mevcut, atlandı: ${TS_ROOT}"
fi

# ============================================================
# 2. AMOS DATASET EĞİTİMİ
# ============================================================
if [ -d "${AMOS_ROOT}" ]; then
    echo ""
    echo "🎓 AMOS dataset eğitimi başlıyor..."
    
    AMOS_OUT="${RUN_DIR}/amos_model"
    mkdir -p "${AMOS_OUT}"
    
    python psoas_ml/amos_train_vfa_pma.py \
        --amos-root "${AMOS_ROOT}" \
        --ts-root "${TS_ROOT}" \
        --c2c-root "${C2C_ROOT}" \
        --use-c2c-override \
        --epochs 60 \
        --batch-size 4 \
        --save-every 10 \
        --out-dir "${AMOS_OUT}"
    
    echo "✅ AMOS eğitimi tamamlandı: ${AMOS_OUT}"
else
    echo "⚠️ AMOS root bulunamadı, AMOS eğitimi atlandı"
fi

# ============================================================
# 3. TBCC DATASET EĞİTİMİ (VARSA)
# ============================================================
TBCC_CSV="${TBCC_ROOT}/ground_truth.csv"
if [ -f "${TBCC_CSV}" ]; then
    echo ""
    echo "🎓 TBCC dataset eğitimi başlıyor..."
    
    TBCC_OUT="${RUN_DIR}/tbcc_model"
    mkdir -p "${TBCC_OUT}"
    
    python psoas_ml/tbcc_train_vfa_pma.py \
        --data-csv "${TBCC_CSV}" \
        --img-root "${TBCC_ROOT}/images" \
        --mask-root "${TBCC_ROOT}/masks" \
        --epochs 60 \
        --batch-size 8 \
        --validate-against-gt \
        --save-every 10 \
        --out-dir "${TBCC_OUT}"
    
    echo "✅ TBCC eğitimi tamamlandı: ${TBCC_OUT}"
else
    echo "⚠️ TBCC ground truth CSV bulunamadı, TBCC eğitimi atlandı"
fi

# ============================================================
# 4. QA OVERLAY ÜRETİMİ
# ============================================================
echo ""
echo "🎨 QA overlay'leri oluşturuluyor..."

QA_OUT="${RUN_DIR}/qa_overlays"
mkdir -p "${QA_OUT}"

# AMOS için overlay
if [ -f "${RUN_DIR}/amos_model/best_model.pth" ]; then
    python tools/generate_improved_overlays.py \
        --model "${RUN_DIR}/amos_model/best_model.pth" \
        --data-root "${AMOS_ROOT}" \
        --out-dir "${QA_OUT}/amos" \
        --max-cases 20
    echo "✅ AMOS overlay üretildi"
fi

# TBCC için overlay (varsa)
if [ -f "${RUN_DIR}/tbcc_model/best_model.pth" ]; then
    python tools/generate_improved_overlays.py \
        --model "${RUN_DIR}/tbcc_model/best_model.pth" \
        --data-root "${TBCC_ROOT}" \
        --out-dir "${QA_OUT}/tbcc" \
        --max-cases 10
    echo "✅ TBCC overlay üretildi"
fi

# ============================================================
# 5. METRİK ANALİZİ VE ÖZET RAPOR
# ============================================================
echo ""
echo "📊 Final metrik analizi yapılıyor..."

SUMMARY_JSON="${RUN_DIR}/summary.json"

cat > "${SUMMARY_JSON}" <<EOF
{
  "run_timestamp": "${TIMESTAMP}",
  "amos_model": "$([ -f ${RUN_DIR}/amos_model/best_model.pth ] && echo 'trained' || echo 'skipped')",
  "tbcc_model": "$([ -f ${RUN_DIR}/tbcc_model/best_model.pth ] && echo 'trained' || echo 'skipped')",
  "qa_overlays": "${QA_OUT}",
  "notes": "Full pipeline run completed on Colab GPU"
}
EOF

echo ""
echo "============================================================"
echo "✅ FULL PIPELINE TAMAMLANDI"
echo "============================================================"
echo "📁 Tüm çıktılar: ${RUN_DIR}"
echo ""
echo "İncelemek için:"
echo "  - Model ağırlıkları: ${RUN_DIR}/*/best_model.pth"
echo "  - QA overlays: ${QA_OUT}"
echo "  - Training logs: ${RUN_DIR}/*/training_log.json"
echo "  - Özet rapor: ${SUMMARY_JSON}"
echo ""
echo "Mac'te Drive senkronize olduktan sonra sonuçları analiz edebilirsin."
echo "============================================================"
