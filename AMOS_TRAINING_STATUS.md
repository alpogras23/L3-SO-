# AMOS Multi-Teacher Training Status

**Güncelleme:** 25 Kasım 2025

## 🎯 Proje Durumu: AKTIF ÇALİŞIYOR

### Tamamlanan Aşamalar

#### 1. TotalSegmentator Model Kurulumu ✅
- **Dataset 952** (abdominal_muscles): 118MB - psoas_major_left/right segmentasyonu
- **Dataset 299** (body): 222MB - torso segmentasyonu  
- **Dataset 297/298** (total): Vertebrae + organs (zaten mevcuttu)
- Kurulum script'i: `scripts/download_ts_weights.sh`

#### 2. İlk 3 Vaka Test Segmentasyonu ✅
- **Vakalar:** amos_0001, amos_0004, amos_0005
- **Task 1:** abdominal_muscles (psoas_major_left/right)
- **Task 2:** total (vertebrae_L3, iliopsoas, organlar)
- **Süre:** ~3-5 saat/vaka (CPU, MacBook)
- **Çıktı:** `data/ts_labels_amos/{case}_abdominal_muscles/`, `{case}_total/`

#### 3. Multi-Label Merge ✅
- **Input:** Per-task NIfTI segmentasyonları
- **Output:** Unified multi-label mask
- **Label IDs:**
  - 29 = vertebrae_L3
  - 55 = psoas_major_left
  - 56 = psoas_major_right
- **Script:** `scripts/ts_merge_all.sh`
- **Dosyalar:** `data/ts_merged_amos/amos_000{1,4,5}_labels.nii.gz`

#### 4. Proof-of-Concept Training (5 Epoch) ✅
- **Dataset:** 61 slices (3 vaka, middle 1/3 region)
- **Model:** MONAI UNet (2D, 1→1, channels: 16,32,64)
- **Loss:** DiceCE
- **Sonuçlar:**
  - val_loss: 0.993
  - train_loss: 1.040
  - Training süresi: ~3 dakika
- **Checkpoint:** `run_mt_mini_nifti/best.ckpt`
- **Script:** `scripts/train_nifti_mini.py`

### 🔄 Devam Eden İşlemler

#### Background Batch Segmentation (PID 30450)
```bash
# Başlatma zamanı: 25 Kas 2025 18:44
# İşlenen vaka: 4/240 (amos_0006 processing)
# Durum: RUNNING
# Log: run_ts_batch_all.log

# İlerleme izleme:
tail -f run_ts_batch_all.log

# Tamamlanan vakalar:
ls data/ts_labels_amos/*_abdominal_muscles/.done | wc -l  # 3/240
ls data/ts_labels_amos/*_total/.done | wc -l              # 3/240
```

**Tahmini Tamamlanma:**
- 237 kalan vaka
- ~3-5 saat/vaka × 2 task = 6-10 saat/vaka
- **Total: 1500-2400 saat** ⚠️
- **Gerçekçi tahmin: 10-15 gün** (CPU-only, sequential)

### 📊 Performans Metrikleri

#### Segmentasyon Hızı (CPU)
- **abdominal_muscles task:**
  - amos_0001: 4:05:32 (270 slices, 768×768)
  - amos_0004: 2:19:45 (160 slices, 512×512)
  - amos_0005: 2:27:58 (160 slices, 768×768)
  - Ortalama: ~52-55 saniye/slice

- **total task (5 models):**
  - amos_0001: 81:51 (5 models × 27 batches)
  - amos_0004: 71:17
  - amos_0005: 83:38
  - Ortalama: ~1.5 saat/vaka

#### Training Hızı (CPU)
- **5 epoch, 61 slices:** ~3 dakika
- **60 epoch, ~5000 slices:** tahmini 6-8 saat

### 🛠️ Oluşturulan Araçlar

1. **scripts/download_ts_weights.sh**
   - TotalSegmentator modellerini manuel indir
   - Resume desteği (`-C -` with curl)
   - Kullanım: `bash scripts/download_ts_weights.sh 952 299`

2. **scripts/ts_batch_segment.sh**
   - Tüm AMOS vakaları için batch segmentasyon
   - Resume logic: `.done` marker ile atlama
   - CPU-friendly: Sequential, tek işlem
   - Kullanım: `bash scripts/ts_batch_segment.sh --input-root DATA --out-root OUT`

3. **scripts/ts_merge_all.sh**
   - Per-task segmentasyonları multi-label'a birleştir
   - Otomatik case detection
   - Kullanım: `bash scripts/ts_merge_all.sh --ts-root data/ts_labels_amos --out-dir data/ts_merged_amos`

4. **scripts/train_nifti_mini.py**
   - NIfTI-based 2D slice training (PNG export bypass)
   - L3 heuristic: middle 1/3 of volume
   - Auto-resize: 256×256 for batching
   - Kullanım: `python scripts/train_nifti_mini.py --nifti-root DATA --ts-merged LABELS --epochs 60`

5. **scripts/build_amos_mini_manifest.sh**
   - CSV manifest builder (eski multiteacher_train için)
   - Not: train_nifti_mini.py artık NIfTI'den direkt okuyor

### 📋 Sonraki Adımlar

#### Kısa Vadeli (1-2 gün)
1. **İlk 10 vakanın tamamlanmasını bekle**
   - Merge test et: `bash scripts/ts_merge_all.sh --cases amos_0001 ... amos_0010`
   - 10-vaka mini training: 10 epoch test
   - QA: overlay görsellerini manuel kontrol et

2. **Hız optimizasyonu değerlendir**
   - GPU erişimi (varsa cloud instance)
   - Paralel processing: 2-3 vaka aynı anda (risk: thermal throttling)
   - `--fast` parametresi test et (düşük kalite trade-off)

#### Orta Vadeli (1 hafta)
1. **İlk 50 vaka ile intermediate training**
   - 30 epoch
   - Val/train split: 80/20
   - Learning curve analizi

2. **Ground truth karşılaştırma**
   - L3_SO_ANALYSIS'deki radyolog ölçümleriyle kıyasla
   - PMA/VFA bias tespiti
   - Model fine-tuning gerekliyse parametreleri ayarla

#### Uzun Vadeli (2-3 hafta)
1. **Full 240-vaka training**
   - 60 epoch
   - Checkpoint her 10 epoch
   - Best model selection

2. **Production deployment**
   - Model export (ONNX/TorchScript)
   - L3_SO_ANALYSIS pipeline entegrasyonu
   - Inference hız optimizasyonu

### ⚠️ Bilinen Kısıtlar

1. **CPU-only Segmentation:**
   - Çok yavaş (10-15 gün tahmini)
   - Thermal throttling riski
   - Paralel processing sınırlı

2. **AMOS GT Limitasyonları:**
   - Organ segmentasyonları var, psoas yok
   - TS outputs pseudo-label olarak kullanılıyor
   - GT weight=0.0 (sadece TS teacher aktif)

3. **L3 Slice Selection:**
   - Heuristic: middle 1/3 of volume
   - Vertebra tespiti yok, manuel slice seçimi gerekebilir
   - Production'da L3 selector entegre edilmeli

### 🔍 Kalite Kontrol Komutları

```bash
# Segmentasyon progress
watch -n 60 "echo 'Abd: $(ls data/ts_labels_amos/*_abdominal_muscles/.done 2>/dev/null | wc -l)/240' && echo 'Total: $(ls data/ts_labels_amos/*_total/.done 2>/dev/null | wc -l)/240'"

# Log monitoring
tail -f run_ts_batch_all.log | grep -E "Case|SKIP|RUN|OK|ERR"

# Merged labels doğrulama
python -c "
import SimpleITK as sitk
import numpy as np
import glob
for p in sorted(glob.glob('data/ts_merged_amos/*_labels.nii.gz'))[:10]:
    arr = sitk.GetArrayFromImage(sitk.ReadImage(p))
    uniq = np.unique(arr[arr>0])
    nonzero = np.count_nonzero(arr)
    print(f'{p.split(\"/\")[-1]:20s} labels={list(uniq)} nonzero={nonzero:6d}')
"

# Disk kullanımı
du -sh data/ts_labels_amos data/ts_merged_amos

# Training checkpoint
ls -lh run_mt_mini_nifti/*.ckpt run_mt_full_60e/*.ckpt 2>/dev/null
```

### 📂 Dizin Yapısı

```
L3_SO_ANALYSIS/
├── data/
│   ├── ts_labels_amos/              # TotalSegmentator raw outputs
│   │   ├── amos_0001_abdominal_muscles/
│   │   │   ├── psoas_major_left.nii.gz
│   │   │   ├── psoas_major_right.nii.gz
│   │   │   └── .done                # Resume marker
│   │   ├── amos_0001_total/
│   │   │   ├── vertebrae_L3.nii.gz
│   │   │   └── .done
│   │   └── ...
│   └── ts_merged_amos/              # Multi-label merged masks
│       ├── amos_0001_labels.nii.gz  # ID: 29,55,56
│       └── ...
├── run_mt_mini_nifti/               # 5-epoch test training
│   ├── best.ckpt
│   ├── logs/
│   └── train.log
├── run_mt_full_60e/                 # Future: full training
├── scripts/
│   ├── download_ts_weights.sh
│   ├── ts_batch_segment.sh
│   ├── ts_merge_all.sh
│   ├── train_nifti_mini.py
│   └── ...
├── run_ts_batch_all.log             # Background batch log
└── AMOS_TRAINING_STATUS.md          # This file
```

### 🚨 Acil Müdahale

**Segmentasyon Durduğunda:**
```bash
# İşlemi durdur
pkill -f TotalSegmentator

# Restart (kaldığı yerden devam eder)
nohup bash scripts/ts_batch_segment.sh \
  --input-root "$HOME/Desktop/amos22/imagesTr" \
  --out-root data/ts_labels_amos \
  > run_ts_batch_all.log 2>&1 &
```

**Disk Dolduğunda:**
```bash
# Segmentasyon outputs sıkıştır
find data/ts_labels_amos -name "*.nii.gz" -exec gzip -9 {} \;

# Eski checkpoint'leri temizle
rm run_mt_mini_nifti/logs/version_*/checkpoints/*.ckpt
```

**Thermal Throttling:**
```bash
# TOTALSEG_ARGS ile throttle ekle
export TOTALSEG_ARGS="--fast"  # Trade-off: hız vs kalite

# Veya manuel pause/resume
pkill -STOP -f TotalSegmentator  # Pause
pkill -CONT -f TotalSegmentator  # Resume
```

---

**Son Güncelleme:** 25 Kasım 2025 18:44  
**Durum:** Background batch segmentation aktif (PID 30450)  
**Sonraki checkpoint:** İlk 10 vaka tamamlandığında
