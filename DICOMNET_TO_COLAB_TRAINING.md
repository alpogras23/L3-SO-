# 🏥 DICOMNET → C2C + TS Teachers → Colab Eğitimi - Komple Rehber

## 📋 İş Akışı Özeti

1. **Masaüstü DICOMNET** klasöründeki aksiyel BT'lerden
2. **Comp2Comp (C2C)** ile VAT/SAT/psoas teacher maskeler üret
3. **TotalSegmentator (TS)** ile vertebra/kas/body teacher maskeler üret
4. **Teachers'ı Google Drive'a** yükle
5. **Colab'da MONAI UNet ile 60 epoch eğitim** yap

---

## 1️⃣ Lokal: C2C Teacher Üretimi

### Gereksinimler:
```bash
# PyDICOM ve NIfTI işlemleri
pip install pydicom nibabel

# dcm2niix (DICOM → NIfTI dönüşümü)
brew install dcm2niix  # macOS

# Comp2Comp (GitHub'dan)
pip install git+https://github.com/StanfordMIMI/Comp2Comp.git
```

### C2C Teacher Oluştur:
```bash
cd ~/Desktop/L3_SO_ANALYSIS
python scripts/generate_c2c_teachers_local.py
```

**Çıktı:** `~/Desktop/C2C_teachers_local/` klasöründe vaka bazında maskeler

---

## 2️⃣ Lokal: TS Teacher Üretimi

### Gereksinimler:
```bash
# TotalSegmentator
pip install TotalSegmentator>=2.3.0
```

### TS Teacher Oluştur:
```bash
cd ~/Desktop/L3_SO_ANALYSIS
python scripts/generate_ts_teachers_from_c2c.py
```

**Çıktı:** `~/Desktop/TS_teachers_local/` klasöründe vaka bazında maskeler

---

## 3️⃣ Teachers'ı Google Drive'a Yükleme

### Manuel Yükleme:
1. `~/Desktop/C2C_teachers_local/` → Google Drive: `C2C_teachers_DICOMNET/`
2. `~/Desktop/TS_teachers_local/` → Google Drive: `TS_teachers_DICOMNET/`
3. `~/Desktop/temp_nifti_for_c2c/` (NIfTI görüntüler) → Google Drive: `DICOMNET_nifti/`

### Otomatik Yükleme (opsiyonel):
```bash
# rclone ile Drive sync
rclone sync ~/Desktop/C2C_teachers_local/ "gdrive:C2C_teachers_DICOMNET/"
rclone sync ~/Desktop/TS_teachers_local/ "gdrive:TS_teachers_DICOMNET/"
rclone sync ~/Desktop/temp_nifti_for_c2c/ "gdrive:DICOMNET_nifti/"
```

---

## 4️⃣ Colab: Eğitim Notebook Hazırlama

### Colab Notebook Aç:
`notebooks/L3_colab_pipeline.ipynb` → Google Colab'da aç

### Veri Yollarını Düzenle (Cell 3):
```python
# DICOM NIfTI görüntüleri
AMOS_IMAGES_ROOT = "/content/drive/MyDrive/DICOMNET_nifti"

# Teacher çıktıları
TS_OUTPUT_ROOT = "/content/drive/MyDrive/TS_teachers_DICOMNET"
C2C_OUTPUT_ROOT = "/content/drive/MyDrive/C2C_teachers_DICOMNET"
```

---

## 5️⃣ Colab: 60 Epoch Eğitim

### Runtime Ayarı:
- **Runtime → Change runtime type → GPU (T4/V100/A100)**

### Hücreleri Sırayla Çalıştır:
1. Drive mount + GPU kontrolü
2. Paket kurulumu (5-10 dk)
3. Veri yolları ayarlama
4. ~~TotalSegmentator inference~~ (Zaten lokal'de yapıldı - atla)
5. ~~Comp2Comp inference~~ (Zaten lokal'de yapıldı - atla)
6. Teacher birleştirme ve manifest
7. Train/Val split
8. MONAI UNet eğitimi (60 epoch - ~6-12 saat)
9. Test inference

---

## 6️⃣ Eğitim Sonrası

### Model İndir:
- `L3_checkpoints/best_model.pt` → Desktop'a indir

### Desktop GUI'de Test:
```bash
cd ~/Desktop/L3_SO_ANALYSIS
python desktop_project/l3_vfa_pma_gui.py --model best_model.pt
```

### Radyolog GT ile Karşılaştır:
```bash
python desktop_project/optimize_params.py \
  --ref ground_truth.csv \
  --cases-dir DICOMNET \
  --model best_model.pt
```

---

## 📊 Beklenen Doğruluk Metrikleri

- **VFA MAE:** < 50 mm²
- **PMA MAE:** < 15 mm²
- **Bias:** ±5% içinde
- **Pearson r:** > 0.90

---

## 🐛 Sorun Giderme

### C2C Teacher Üretimi Başarısız:
- `dcm2niix` kurulu mu kontrol et: `dcm2niix --version`
- Comp2Comp kurulumu: `pip install git+https://github.com/StanfordMIMI/Comp2Comp.git`
- DICOM dosyaları bozuk olabilir: manuel kontrol

### TS Teacher Üretimi Başarısız:
- TotalSegmentator versiyon: `pip install --upgrade TotalSegmentator>=2.3.0`
- GPU yetersizse: `--fast` bayrağını kaldır (daha yavaş ama düşük RAM)

### Colab Eğitim Hatası:
- Drive kotası dolmuş olabilir (teacher mask'ler büyük)
- GPU runtime kesilmişse: Runtime → Reconnect
- OOM (Out of Memory): Batch size'ı düşür (4 → 2)

---

## 📂 Klasör Yapısı

```
~/Desktop/
├── DICOMNET/                       # Orijinal DICOM BT'ler
│   ├── 2.000000-Body 3.0 CE-06674/
│   ├── 2.000000-Body 3.0 CE-31390/
│   └── ...
├── C2C_teachers_local/             # Comp2Comp çıktıları
│   ├── 2.000000-Body 3.0 CE-06674/
│   │   ├── vat.nii.gz
│   │   ├── sat.nii.gz
│   │   └── psoas.nii.gz
│   └── ...
├── TS_teachers_local/              # TotalSegmentator çıktıları
│   ├── 2.000000-Body 3.0 CE-06674/
│   │   ├── vertebrae_L3.nii.gz
│   │   ├── autochthon_left.nii.gz
│   │   └── ...
│   └── ...
└── temp_nifti_for_c2c/            # Ara NIfTI dosyaları
    ├── 2.000000-Body 3.0 CE-06674.nii.gz
    └── ...

Google Drive:
├── DICOMNET_nifti/                # NIfTI görüntüler (eğitim için)
├── C2C_teachers_DICOMNET/         # C2C maskeler
├── TS_teachers_DICOMNET/          # TS maskeler
├── L3_checkpoints/                # Eğitim model checkpoint'leri
└── L3_test_outputs/               # Test sonuçları
```

---

## ✅ Checklist

- [ ] Lokal: C2C teacher üretimi tamamlandı
- [ ] Lokal: TS teacher üretimi tamamlandı
- [ ] Teachers Google Drive'a yüklendi
- [ ] Colab notebook veri yolları güncellendi
- [ ] Colab GPU runtime seçildi
- [ ] 60 epoch eğitim tamamlandı
- [ ] Best model indirildi
- [ ] Desktop GUI'de test edildi
- [ ] Radyolog GT ile validasyon yapıldı

---

## 📞 Destek

Herhangi bir sorun için:
- Script hatalarını `qa_out/` klasörüne log'la
- Colab hataları için cell çıktısını kaydet
- Teacher üretim hatalarını `c2c_manifest.json` / `ts_manifest.json` ile kontrol et
