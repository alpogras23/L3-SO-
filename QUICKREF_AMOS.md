# AMOS Multi-Teacher Training - Quick Reference

**Durum:** Background batch segmentation aktif (25 Kas 2025, 19:30)

## 🚀 En Çok Kullanılan Komutlar

### 1. Status Kontrolü
```bash
bash scripts/check_amos_status.sh          # Dashboard görünümü
tail -f run_ts_batch_all.log               # Live log
watch -n 60 bash scripts/check_amos_status.sh  # Auto-refresh
```

### 2. Segmentasyon Yönetimi
```bash
# Background batch (sequential, devam ediyor)
nohup bash scripts/ts_batch_segment.sh \
  --input-root ~/Desktop/amos22/imagesTr \
  --out-root data/ts_labels_amos \
  > run_ts_batch_all.log 2>&1 &

# Parallel batch (hızlı ama thermal risk)
nohup bash scripts/ts_batch_parallel.sh \
  --input-root ~/Desktop/amos22/imagesTr \
  --out-root data/ts_labels_amos \
  --parallel 2 \
  > run_ts_batch_parallel.log 2>&1 &

# Pause/Resume
pkill -STOP -f TotalSegmentator  # Pause
pkill -CONT -f TotalSegmentator  # Resume
```

### 3. Label Merging
```bash
# Tüm tamamlanan vakalar
bash scripts/ts_merge_all.sh \
  --ts-root data/ts_labels_amos \
  --out-dir data/ts_merged_amos

# Belirli vakalar
bash scripts/ts_merge_all.sh \
  --ts-root data/ts_labels_amos \
  --out-dir data/ts_merged_amos \
  --cases amos_0001 amos_0004 amos_0005
```

### 4. Model Eğitimi
```bash
# Manuel training (belirli epoch count)
python scripts/train_nifti_mini.py \
  --nifti-root ~/Desktop/amos22/imagesTr \
  --ts-merged data/ts_merged_amos \
  --epochs 10 \
  --batch-size 2 \
  --threads 1 \
  --out run_mt_10ep

# Otomatik checkpoint training (background)
nohup bash scripts/auto_train_checkpoints.sh \
  --watch-interval 3600 \
  > auto_train_checkpoints.log 2>&1 &
```

### 5. Training Analizi
```bash
# Loss curves ve summary
python scripts/visualize_training.py \
  --runs run_mt_mini_nifti run_mt_10cases_10ep \
  --out training_plots

# Model export
python scripts/export_model.py \
  --checkpoint run_mt_full_60e/best.ckpt \
  --format onnx \
  --test
```

## 📊 Checkpoint Milestones

| Vakalar | Epochs | Tahmini Süre | Auto Training | Çıktı Dizini |
|---------|--------|--------------|---------------|--------------|
| 3 (✅) | 5 | 3 dk | ✅ Tamamlandı | `run_mt_mini_nifti/` |
| 10 | 10 | ~15 dk | 🔄 Bekliyor | `run_mt_10cases_10ep/` |
| 25 | 20 | ~45 dk | 🔄 Bekliyor | `run_mt_25cases_20ep/` |
| 50 | 30 | ~2 saat | 🔄 Bekliyor | `run_mt_50cases_30ep/` |
| 100 | 40 | ~4 saat | 🔄 Bekliyor | `run_mt_100cases_40ep/` |
| 200 | 50 | ~7 saat | 🔄 Bekliyor | `run_mt_200cases_50ep/` |
| 240 | 60 | ~8 saat | 🔄 Bekliyor | `run_mt_full_240cases_60ep/` |

## 🛠️ Script Referansı

### Core Scripts
- `download_ts_weights.sh` - Model indirme (952, 299)
- `ts_batch_segment.sh` - Sequential batch segmentation (resumable)
- `ts_batch_parallel.sh` - Parallel segmentation (2-3 jobs)
- `ts_merge_all.sh` - Multi-label merge
- `train_nifti_mini.py` - Training engine
- `check_amos_status.sh` - Status dashboard

### Automation
- `auto_train_checkpoints.sh` - Otomatik checkpoint training (10,25,50,100,200,240 vakalar)

### Analysis
- `visualize_training.py` - Loss curves, learning rate plots, summary
- `export_model.py` - ONNX/TorchScript/StateDict export

## ⚡ Hızlı Troubleshooting

### Problem: Segmentasyon çok yavaş
```bash
# Parallel mode kullan (thermal risk!)
bash scripts/ts_batch_parallel.sh --parallel 2

# Veya --fast flag ekle (kalite trade-off)
export TOTALSEG_ARGS="--fast"
bash scripts/ts_batch_segment.sh ...
```

### Problem: İşlem durdu
```bash
# PID kontrolü
ps aux | grep -E "TotalSegmentator|ts_batch" | grep -v grep

# Restart (resume logic sayesinde kaldığı yerden devam eder)
nohup bash scripts/ts_batch_segment.sh \
  --input-root ~/Desktop/amos22/imagesTr \
  --out-root data/ts_labels_amos \
  >> run_ts_batch_all.log 2>&1 &
```

### Problem: Disk doldu
```bash
# Kullanımı kontrol et
du -sh data/ts_labels_amos data/ts_merged_amos

# Intermediate files temizle (UYARI: yeniden segmentasyon gerekebilir!)
find data/ts_labels_amos -type f ! -name ".done" -delete

# Veya compress
find data/ts_labels_amos -name "*.nii.gz" -exec gzip -9 {} \;
```

### Problem: Training crash
```bash
# Batch size düşür
python scripts/train_nifti_mini.py ... --batch-size 1

# Thread limit
--threads 1

# Memory check
top -l 1 | grep PhysMem
```

## 📁 Önemli Dizinler

```
L3_SO_ANALYSIS/
├── data/
│   ├── ts_labels_amos/          # TS raw outputs (~35-40 GB)
│   └── ts_merged_amos/          # Multi-label merged (~60-80 MB)
├── run_mt_*/                    # Training runs
├── training_plots/              # Visualization outputs
├── scripts/                     # Tüm script'ler
├── run_ts_batch_all.log         # Batch segmentation log
└── auto_train_checkpoints.log   # Auto training log
```

## 📈 Beklenen Sonuçlar

### Segmentasyon (CPU)
- **Hız:** ~3-5 saat/vaka (abd + total)
- **Toplam:** 240 vaka × 4 saat = ~960 saat = **10-15 gün**

### Training
- **POC (5 epoch):** 3 dk ✅
- **10 cases (10 epoch):** ~15 dk
- **Full (60 epoch, 240 cases):** ~8 saat

### Model Performance (beklenen)
- **Val Loss:** <0.5 (DiceCE)
- **Dice Score:** >0.85 (psoas segmentation)
- **Inference:** ~50-100 ms/slice (CPU)

## 🎯 Güncel Todo List

- [x] Model setup ve test segmentation
- [x] POC training (5 epoch, 3 vaka)
- [x] Batch segmentation başlat
- [x] Monitoring ve automation tools
- [ ] 10-case checkpoint (~2-3 gün)
- [ ] 50-case intermediate training (~1 hafta)
- [ ] Full 240-case production training (~2 hafta)

## 🔗 Dokümantasyon

- **Kapsamlı Kılavuz:** `README_AMOS_MULTITEACHER.md`
- **Detaylı Durum:** `AMOS_TRAINING_STATUS.md`
- **Bu Döküman:** `QUICKREF_AMOS.md`

---

**Son Güncelleme:** 25 Kasım 2025 19:30  
**Background Process:** PID 30450 (batch segmentation)  
**Current Case:** amos_0006 (14% → 39/270 slices)  
**Komut:** `bash scripts/check_amos_status.sh`
