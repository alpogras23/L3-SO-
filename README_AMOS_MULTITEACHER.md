# AMOS + TotalSegmentator Multi-Teacher Training

**Güncel Durum:** Background batch segmentation aktif (25 Kas 2025)

Bu proje, AMOS22 CT dataset'i üzerinde TotalSegmentator pseudo-labels kullanarak multi-teacher psoas/vertebra segmentation modeli eğitimini gerçekleştirir.

## 🎯 Proje Özeti

**Amaç:** L3 vertebra seviyesinde psoas kasları ve vertebra segmentasyonu için UNet model eğitimi

**Dataset:** AMOS22 (240 CT training vakaları)
- **GT:** AMOS organ segmentasyonları (psoas yok, sadece organ labels)
- **Pseudo-labels:** TotalSegmentator v2.11.0 outputs
  - Task 952 (abdominal_muscles): psoas_major_left/right
  - Task 297 (total): vertebrae_L3

**Model:** MONAI UNet 2D (1→1, channels: 16,32,64)

**Loss:** DiceCELoss (sigmoid, squared_pred)

## 🚀 Hızlı Başlangıç

### Ön Koşullar

```bash
# Python environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Dependencies
pip install torch torchvision pytorch-lightning
pip install monai SimpleITK opencv-python
pip install TotalSegmentator
```

### 1. TotalSegmentator Modellerini İndir

```bash
# Manuel indirme (internet yavaşsa)
bash scripts/download_ts_weights.sh 952 299

# Veya otomatik (ilk segmentasyonda indirilir)
```

### 2. AMOS Vakalarını Segmente Et

```bash
# İlk 3 vaka ile test (proof-of-concept)
bash scripts/ts_batch_segment.sh \
  --input-root ~/Desktop/amos22/imagesTr \
  --out-root data/ts_labels_amos \
  --max-cases 3

# Tüm 240 vaka (background, ~10-15 gün CPU)
nohup bash scripts/ts_batch_segment.sh \
  --input-root ~/Desktop/amos22/imagesTr \
  --out-root data/ts_labels_amos \
  > run_ts_batch_all.log 2>&1 &

# İlerleme izleme
tail -f run_ts_batch_all.log
bash scripts/check_amos_status.sh
```

### 3. Segmentasyonları Birleştir (Merge)

```bash
# İlk 3 vaka
bash scripts/ts_merge_all.sh \
  --ts-root data/ts_labels_amos \
  --out-dir data/ts_merged_amos \
  --cases amos_0001 amos_0004 amos_0005

# Tüm tamamlanan vakalar
bash scripts/ts_merge_all.sh \
  --ts-root data/ts_labels_amos \
  --out-dir data/ts_merged_amos
```

### 4. Model Eğitimi

```bash
# Test training (5 epoch, 3 vaka)
python scripts/train_nifti_mini.py \
  --nifti-root ~/Desktop/amos22/imagesTr \
  --ts-merged data/ts_merged_amos \
  --cases amos_0001 amos_0004 amos_0005 \
  --epochs 5 \
  --batch-size 2 \
  --threads 1 \
  --out run_mt_mini_nifti

# Full training (60 epoch, tüm vakalar)
python scripts/train_nifti_mini.py \
  --nifti-root ~/Desktop/amos22/imagesTr \
  --ts-merged data/ts_merged_amos \
  --epochs 60 \
  --batch-size 1 \
  --threads 1 \
  --out run_mt_full_60e
```

## 📂 Dosya Yapısı

```
L3_SO_ANALYSIS/
├── data/
│   ├── ts_labels_amos/              # TS raw outputs
│   │   ├── amos_0001_abdominal_muscles/
│   │   │   ├── psoas_major_left.nii.gz
│   │   │   ├── psoas_major_right.nii.gz
│   │   │   └── .done
│   │   └── amos_0001_total/
│   │       ├── vertebrae_L3.nii.gz
│   │       └── .done
│   └── ts_merged_amos/              # Multi-label merged
│       └── amos_0001_labels.nii.gz  # IDs: 29,55,56
├── scripts/
│   ├── download_ts_weights.sh       # Model downloader
│   ├── ts_batch_segment.sh          # Batch segmentation
│   ├── ts_merge_all.sh              # Label merging
│   ├── train_nifti_mini.py          # Training script
│   └── check_amos_status.sh         # Status checker
├── run_mt_mini_nifti/               # Test training output
│   ├── best.ckpt
│   └── logs/
├── run_ts_batch_all.log             # Batch log
└── AMOS_TRAINING_STATUS.md          # Detailed status
```

## 🛠️ Script Referansı

### download_ts_weights.sh
Manuel TotalSegmentator model indirme (internet yavaşsa).

```bash
bash scripts/download_ts_weights.sh 952 299 [--force]
```

**Parametreler:**
- `952`: abdominal_muscles task (psoas)
- `299`: body task (torso)
- `--force`: Mevcut indirmeleri override et

### ts_batch_segment.sh
Batch TotalSegmentator inference (resumable).

```bash
bash scripts/ts_batch_segment.sh \
  --input-root /path/to/imagesTr \
  --out-root data/ts_labels_amos \
  [--tasks "abdominal_muscles total"] \
  [--max-cases N]
```

**Özellikleri:**
- Resume logic: `.done` marker ile atlanır
- Sequential execution (CPU-friendly)
- Progress tracking

### ts_merge_all.sh
Per-task outputs → multi-label NIfTI.

```bash
bash scripts/ts_merge_all.sh \
  --ts-root data/ts_labels_amos \
  --out-dir data/ts_merged_amos \
  [--cases amos_0001 amos_0002 ...] \
  [--include-fat]
```

**Label ID Mapping:**
- 29 = vertebrae_L3
- 55 = psoas_major_left
- 56 = psoas_major_right
- 100 = torso_fat (eğer `--include-fat`)
- 101 = subcutaneous_fat (eğer `--include-fat`)

### train_nifti_mini.py
NIfTI-based 2D slice training.

```bash
python scripts/train_nifti_mini.py \
  --nifti-root /path/to/imagesTr \
  --ts-merged /path/to/merged_labels \
  [--cases amos_0001 amos_0002 ...] \
  [--epochs 60] \
  [--batch-size 2] \
  [--lr 1e-3] \
  [--threads 1] \
  [--out run_output]
```

**Özellikleri:**
- L3 heuristic: middle 1/3 of volume
- Auto-resize: 256×256 for batching
- CPU-safe: configurable threads
- Val/train split: 80/20

### check_amos_status.sh
Pipeline status dashboard.

```bash
bash scripts/check_amos_status.sh
```

**Çıktı:**
- Background process status
- Segmentation progress (abd/total)
- Merged labels count
- Training checkpoints
- Disk usage
- Next steps suggestion

## 📊 Performans Beklentileri

### Segmentasyon (CPU, MacBook M-series)
- **abdominal_muscles:** ~2-4 saat/vaka
- **total (5 models):** ~1-1.5 saat/vaka
- **Toplam:** ~3-5.5 saat/vaka
- **240 vaka:** ~10-15 gün

### Training (CPU)
- **5 epoch, 61 slices:** ~3 dakika
- **60 epoch, ~5000 slices:** ~6-8 saat

### Disk Kullanımı
- **TS labels (240 vaka):** ~35-40 GB
- **Merged labels:** ~60-80 MB
- **Training checkpoints:** ~1-2 GB

## 🎓 Training Best Practices

### CPU Thermal Management
```bash
# Low batch size
--batch-size 1

# Thread limit
--threads 1

# Accumulate gradients
# (eski multiteacher_train.py'de: accumulate_grad_batches=4)
```

### Checkpoint Strategy
```bash
# Her 10 epoch checkpoint
# ModelCheckpoint monitor='val_loss'
# save_top_k=3
```

### Learning Rate Schedule
```bash
# Initial: 1e-3
# ReduceLROnPlateau: patience=5, factor=0.5
# Min: 1e-6
```

## 🔍 Kalite Kontrol

### Segmentasyon Doğrulama
```python
import SimpleITK as sitk
import numpy as np

# Merged label kontrolü
img = sitk.ReadImage('data/ts_merged_amos/amos_0001_labels.nii.gz')
arr = sitk.GetArrayFromImage(img)
unique_labels = np.unique(arr[arr > 0])
print(f"Labels: {unique_labels}")  # Beklenen: [29, 55, 56]

# Nonzero pixel count
for label_id in unique_labels:
    count = np.count_nonzero(arr == label_id)
    print(f"Label {label_id}: {count} pixels")
```

### Training Curve Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

# CSV logger outputs
df = pd.read_csv('run_mt_mini_nifti/logs/version_0/metrics.csv')
plt.plot(df['epoch'], df['train_loss'], label='train')
plt.plot(df['epoch'], df['val_loss'], label='val')
plt.legend()
plt.savefig('training_curve.png')
```

## ⚠️ Bilinen Sorunlar ve Çözümler

### 1. TotalSegmentator Model İndirme Yavaş
**Sorun:** Model indirme 100-200 KB/s hızında takılıyor.

**Çözüm:**
```bash
# Manuel indirme script kullan
bash scripts/download_ts_weights.sh 952 299

# Veya tarayıcıda indir + manuel extract:
# ~/.totalsegmentator/nnunet/results/ altına kopyala
```

### 2. Batch İşlem Durdu
**Sorun:** Sistem uyku moduna girdi veya işlem kesildi.

**Çözüm:**
```bash
# Resume (kaldığı yerden devam eder)
nohup bash scripts/ts_batch_segment.sh \
  --input-root ~/Desktop/amos22/imagesTr \
  --out-root data/ts_labels_amos \
  >> run_ts_batch_all.log 2>&1 &
```

### 3. Disk Doldu
**Sorun:** TS outputs çok yer kaplıyor.

**Çözüm:**
```bash
# Intermediate files temizle (sadece merged labels tut)
# UYARI: Segmentasyonu tekrar çalıştırmak gerekirse zararlı!
rm -rf data/ts_labels_amos/amos_*_abdominal_muscles/
rm -rf data/ts_labels_amos/amos_*_total/

# Veya compress:
find data/ts_labels_amos -name "*.nii.gz" -exec gzip -9 {} \;
```

### 4. AMOS Farklı Boyutlar (Batching Error)
**Sorun:** `RuntimeError: stack expects each tensor to be equal size`

**Çözüm:** train_nifti_mini.py otomatik resize yapar (256×256). Eğer hala sorun varsa `--batch-size 1` kullan.

## 🔗 Entegrasyon

### L3_SO_ANALYSIS Pipeline ile Birleştirme

```python
# core_mini.py veya l3_vfa_pma_core_v3_7.py'ye psoas segmentation ekle

from scripts.train_nifti_mini import LitModel
import torch

# Load trained model
ckpt_path = 'run_mt_full_60e/best.ckpt'
model = LitModel.load_from_checkpoint(ckpt_path)
model.eval()

# Inference on L3 slice
def segment_psoas_l3(hu_slice: np.ndarray) -> np.ndarray:
    """
    Args:
        hu_slice: (H, W) HU values
    Returns:
        psoas_mask: (H, W) binary mask
    """
    # Normalize
    img_norm = (hu_slice - hu_slice.min()) / (hu_slice.max() - hu_slice.min() + 1e-6)
    img_t = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
    
    # Inference
    with torch.no_grad():
        pred = model(img_t)  # (1,1,H,W)
        pred = torch.sigmoid(pred)
        mask = (pred > 0.5).squeeze().numpy()
    
    return mask
```

## 📚 Referanslar

- **TotalSegmentator:** https://github.com/wasserth/TotalSegmentator
- **AMOS22:** https://amos22.grand-challenge.org/
- **MONAI:** https://monai.io/
- **PyTorch Lightning:** https://lightning.ai/

## 📄 Lisans

Bu proje L3_SO_ANALYSIS projesiyle aynı lisans altındadır.

---

**Son Güncelleme:** 25 Kasım 2025  
**Durum:** Background batch segmentation aktif  
**Detaylı Status:** `AMOS_TRAINING_STATUS.md`  
**Hızlı Kontrol:** `bash scripts/check_amos_status.sh`
