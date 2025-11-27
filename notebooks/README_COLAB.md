# Colab Pro+ L3 VFA/PMA Eğitim Notebook'u

## Hızlı Başlangıç

### 1. Colab'da Açın

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/notebooks/colab_pro_plus_L3_training.ipynb)

> Not: Yukarıdaki badge için GitHub repo linkinizi güncelleyin, veya doğrudan Colab'a yükleyin.

### 2. Veri Hazırlığı

AMOS22 verilerinizi Google Drive'a yerleştirin:

```
/content/drive/MyDrive/AMOS22/
├── imagesTr/           # CT hacimleri (.nii.gz)
│   ├── amos_0001.nii.gz
│   ├── amos_0002.nii.gz
│   └── ...
├── imagesTs/           # Test vakaları (opsiyonel)
└── c2c_labels/         # C2C maskeleri (opsiyonel)
    ├── amos_0001/
    │   ├── psoas_left.nii.gz
    │   └── psoas_right.nii.gz
    └── ...
```

### 3. CONFIG Ayarları

Notebook'un ilk hücresinde CONFIG parametrelerini düzenleyin:

```python
CONFIG = {
    'USE_DRIVE': True,
    'DRIVE_DATA_ROOT': '/content/drive/MyDrive/AMOS22',
    'MAX_CASES': 10,              # İlk denemede 10 vaka ile test edin
    'MERGE_STRATEGY': 'TS_ONLY',  # veya 'UNION' / 'INTERSECT'
    'MAX_EPOCHS': 10,             # İlk denemede 10 epoch, sonra 60'a çıkarın
    'USE_AMP': True               # GPU hızlandırma (önerilir)
}
```

### 4. Çalıştırın

Colab menüsünden: **Runtime → Run all**

## Akış Özeti

1. **Ortam kurulumu** → PyTorch, MONAI, TotalSegmentator kurulur
2. **GPU kontrolü** → CUDA durumu kontrol edilir
3. **Drive mount** → AMOS22 verileri IMAGES_DIR'e symlink/copy
4. **Teacher üretimi** → TotalSegmentator ile psoas ve L3 segmentasyonu
5. **Birleştirme** → TS + C2C (opsiyonel) maskeler birleştirilir
6. **Dataset** → 3D hacimlerden 2D orta dilim + 256×256 resize
7. **Eğitim** → 60 epoch (veya CONFIG'te belirlenen), AMP'li GPU eğitimi
8. **Overlay** → Segmentasyon sonuçlarını görselleştirme
9. **Export** → Checkpoint'ler Drive'a kaydedilir

## Özellikler

### ✅ Tek Tık Çalışma
- Runtime → Run all ile tüm pipeline otomatik çalışır
- Veri olmasa bile hata vermez (güvenli boş-dataset kontrolü)

### ✅ Drive Entegrasyonu
- AMOS22 verilerini otomatik içe aktarır (symlink veya copy)
- Checkpoint'leri otomatik olarak Drive'a yedekler

### ✅ TotalSegmentator Teacher
- `abdominal_muscles` task → psoas major left/right
- `total` task → vertebrae_L3 (hızlandırma: --fast + torso ROI)
- İdempotent: önceden üretilmiş maskeler atlanır

### ✅ C2C Entegrasyon
- Opsiyonel ikinci teacher desteği
- Merge stratejileri: TS_ONLY / C2C_ONLY / UNION / INTERSECT

### ✅ 2D Eğitim (3D Hacimlerden)
- Orta dilim otomatik seçilir
- 256×256 boyut standardizasyonu
- HU normalizasyonu (-1000 to 1000)

### ✅ GPU Optimizasyonları
- Mixed Precision Training (AMP) → 2-3x hız
- pin_memory, persistent_workers
- Batch size otomatik ayar (GPU varsa 4, yoksa 2)

### ✅ Augmentasyonlar
- Random flip, rotate, zoom
- Sadece training için (validation temiz)

### ✅ Checkpoint Yönetimi
- En iyi validation loss → best.pt
- Son epoch → last.pt
- TorchScript export → model_scripted.pt

## Çıktılar

### Eğitim Sırasında
- Epoch başına train/val loss
- En iyi checkpoint kayıt bildirimleri

### Eğitim Sonrası
- `/content/outputs/checkpoints/` altında best.pt, last.pt
- Drive'da `/MyDrive/L3_VFA_PMA_runs/model_export_YYYYMMDD-HHMMSS/`
- Overlay görselleştirmesi (matplotlib figure)

## Önerilen Çalışma Sırası

### İlk Deneme (Hızlı Test)
```python
CONFIG = {
    'MAX_CASES': 3,
    'MAX_EPOCHS': 5,
    ...
}
```
→ ~10-15 dakika (GPU T4)

### Küçük Eğitim
```python
CONFIG = {
    'MAX_CASES': 10,
    'MAX_EPOCHS': 20,
    ...
}
```
→ ~30-60 dakika

### Tam Eğitim
```python
CONFIG = {
    'MAX_CASES': None,  # Tüm veriler
    'MAX_EPOCHS': 60,
    ...
}
```
→ AMOS22 (240 vaka) için ~6-10 saat (Colab Pro+ GPU)

## Sorun Giderme

### "Dataset empty" mesajı
- Drive'da AMOS22 klasörünün doğru yolda olduğundan emin olun
- `DRIVE_DATA_ROOT` ve `DRIVE_IMAGES_SUBDIRS` CONFIG'i kontrol edin
- 4. hücre çıktısında "Bulunan görüntü adedi" satırını inceleyin

### TotalSegmentator yavaş
- GPU runtime seçildiğinden emin olun (Runtime → Change runtime type)
- İlk çalışmada TS modelleri indirilir (~2GB), sonraki çalışmalarda cache'den hızlı

### Memory hatası
- `BATCH_SIZE` düşürün (ör. 2 veya 1)
- `MAX_CASES` ile vaka sayısını azaltın

### Symlink hatası
- `COPY_FROM_DRIVE: 'copy'` yapın (daha yavaş ama güvenli)

## Teknik Detaylar

### Model Mimarisi
- MONAI 2D UNet
- 1 kanal giriş (CT), 2 kanal çıkış (left/right psoas)
- channels=(16, 32, 64), strides=(2, 2)

### Loss Fonksiyonu
- DiceCELoss (Dice + Cross-Entropy)
- softmax aktivasyon, one-hot target

### Optimizer
- AdamW, lr=1e-3
- Scheduler yok (basit eğitim)

### Transform Pipeline
```
3D NIfTI → Orta dilim → HU [-1000,1000] → [0,1] → Resize 256×256 → Augment → Tensor
```

## Katkıda Bulunma

Geliştirme önerileri:
- [ ] Multi-slice (orta ±2 dilim) desteği
- [ ] Learning rate scheduler
- [ ] Early stopping
- [ ] Cross-validation split
- [ ] Dice score metriği
- [ ] TensorBoard logging

## Lisans

Bu notebook, L3 VFA/PMA projesi kapsamında geliştirilmiştir.
