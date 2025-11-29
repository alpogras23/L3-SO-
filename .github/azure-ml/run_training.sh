#!/bin/bash
set -e

echo "🚀 Azure ML L3 Training Pipeline Başlatılıyor..."
echo "================================================"

# Ortam değişkenleri
EPOCHS=${EPOCHS:-60}
SITE_PRESET=${SITE_PRESET:-eval}
RUN_TS=${RUN_TS:-true}
RUN_C2C=${RUN_C2C:-false}
AMOS22_ROOT=${AMOS22_ROOT:-/mnt/data/amos22}
DICOM_OUT=${DICOM_OUT:-./outputs/dicom}
TEACHERS_OUT=${TEACHERS_OUT:-./outputs/teachers}
QA_OUT=${QA_OUT:-./qa_out}
CKPT_DIR=${CKPT_DIR:-./ckpts}

echo "Parametreler:"
echo "  EPOCHS: $EPOCHS"
echo "  SITE_PRESET: $SITE_PRESET"
echo "  RUN_TS: $RUN_TS"
echo "  RUN_C2C: $RUN_C2C"
echo "  AMOS22_ROOT: $AMOS22_ROOT"

# Dizinler oluştur
mkdir -p "$DICOM_OUT" "$TEACHERS_OUT" "$QA_OUT" "$CKPT_DIR"

# GPU kontrolü
echo ""
echo "🎮 GPU Kontrolü:"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '- Devices:', torch.cuda.device_count())"

# 1. AMOS22 -> DICOM dönüşümü
echo ""
echo "📦 AMOS22 NIfTI -> DICOM dönüşümü..."
python scripts/prepare_amos_dicom.py \
  --amos-root "$AMOS22_ROOT" \
  --out "$DICOM_OUT" \
  --prefer-plastimatch false

# 2. Öğretmen çıkarımı (TS + C2C)
echo ""
echo "🧠 Öğretmen çıkarımı (TotalSegmentator + C2C)..."
python scripts/run_teachers.py \
  --dicom-root "$DICOM_OUT" \
  --out "$TEACHERS_OUT" \
  --labels-cfg scripts/labels_config_example.json \
  --use-ts "$RUN_TS" \
  --use-c2c "$RUN_C2C"

# 3. Preset ayarı ve settings.json override
echo ""
echo "⚙️  Preset ve parametre ayarları..."
python scripts/select_preset.py "$SITE_PRESET"

# settings.json acil doğruluk override'ları
cat > settings.json <<EOF
{
  "VB_HU_MIN": 150,
  "VB_HU_MAX": 4000,
  "FASCIA_RING_BAND_MM": 20,
  "GUARD_BAND_MM": 30,
  "VFA_HU_LOW": -180,
  "VFA_HU_HIGH": -20,
  "PSOAS_HU_MIN": -10,
  "PSOAS_HU_MAX": 100,
  "OVERLAY_ALPHA": 0.75,
  "OVERLAY_NORMALIZE_HU": true,
  "OVERLAY_SHOW_CONFIDENCE": true,
  "OVERLAY_SHOW_MEASUREMENTS": true,
  "OVERLAY_VERTEBRA_OUTLINE": true
}
EOF

# 4. L3 seçimi, fasya, psoas ve VFA/PMA hesapları
echo ""
echo "🔍 L3 seçimi ve VFA/PMA hesaplaması..."
python scripts/run_amos_multiteacher_pipeline.py \
  --dicom-root "$DICOM_OUT" \
  --teachers-root "$TEACHERS_OUT" \
  --qa-out "$QA_OUT" \
  --labels-cfg scripts/labels_config_example.json \
  --save-overlays \
  --overlay-high-quality

# Manifest konumunu bul
TRAIN_CSV=""
for candidate in \
  "$TEACHERS_OUT/manifest.csv" \
  "$TEACHERS_OUT/train_manifest.csv" \
  "./train_manifest.csv"; do
  if [ -f "$candidate" ]; then
    TRAIN_CSV="$candidate"
    break
  fi
done

if [ -z "$TRAIN_CSV" ]; then
  echo "❌ HATA: Training manifest bulunamadı!"
  exit 1
fi
echo "✅ Manifest bulundu: $TRAIN_CSV"

# 5. 60 epok eğitim
echo ""
echo "🏋️  $EPOCHS epok eğitim başlatılıyor..."
python -m psoas_ml.multiteacher_train \
  --csv "$TRAIN_CSV" \
  --img_root "$DICOM_OUT" \
  --mask_root "$TEACHERS_OUT" \
  --epochs "$EPOCHS" \
  --ckpt_dir "$CKPT_DIR"

# 6. Eğitim görselleştirme
echo ""
echo "📊 Eğitim görselleştirme ve özet..."
python scripts/visualize_training.py \
  --log_dir "$CKPT_DIR" \
  --out "$CKPT_DIR/training_summary.json"

# 7. Parametre optimizasyonu (ground truth varsa)
if [ -f "/mnt/data/ground_truth.csv" ]; then
  echo ""
  echo "🎯 Ground truth ile parametre optimizasyonu..."
  python desktop_project/optimize_params.py \
    --ref /mnt/data/ground_truth.csv \
    --cases-dir "$DICOM_OUT" \
    --random 50 \
    --bias-penalty 0.2 \
    --pearson-weight 0.1
fi

echo ""
echo "✅ Pipeline tamamlandı!"
echo "Çıktılar:"
echo "  - DICOM: $DICOM_OUT"
echo "  - Öğretmen maskeler: $TEACHERS_OUT"
echo "  - QA overlayler: $QA_OUT"
echo "  - Model checkpoints: $CKPT_DIR"
