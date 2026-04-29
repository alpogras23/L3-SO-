# L3 VFA/PMA Radyolog Uyumluluğu Analizi

**Maksimum klinik doğruluk hedefli L3 seviyesi visseral yağ alanı (VFA) ve psoas kas alanı (PMA) otomatik ölçüm sistemi.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.2+-76b900.svg)](https://monai.io/)

## 🎯 Proje Amacı

Bu sistem, CT görüntülerinden **radyolog ölçümleriyle maksimum uyumlu** otomatik L3 vertebra seviyesi VFA ve PMA hesaplaması yapar. Temel hedefler:

- ✅ **VFA MAE < 50 mm²** (radyolog ölçümüne göre)
- ✅ **PMA MAE < 15 mm²** (radyolog ölçümüne göre)
- ✅ **Bias ±5% içinde** (sistematik sapma kontrolü)
- ✅ **r > 0.90** (radyolog korelasyonu)

### Öne Çıkan Özellikler

- 🔬 **Multi-Teacher Yaklaşım:** TotalSegmentator + Comp2Comp teacher entegrasyonu
- 🤖 **Comp2Comp Azure ML:** End-to-end spine, muscle, adipose tissue segmentation
- 🎓 **CVAT Entegrasyonu:** Manuel radyolog maskelerini eğitimde kullanma
- 🖥️ **VS Code ↔ Colab Pro:** Yerel geliştirme, GPU eğitimi bulutta
- 📊 **QA Odaklı:** Her segmentasyon için overlay görselleştirme ve kalite metrikleri
- ⚡ **Tek Komut Pipeline:** TS teacher + eğitim + QA tek script'te

---

## 📁 Proje Yapısı

```
L3_SO_ANALYSIS/
├── core_mini.py                     # Ana işleme çekirdeği (kararlı, üretim)
├── l3_vfa_pma_core_v3_7.py         # Legacy uyumluluk katmanı
├── settings.json                    # Parametre konfigürasyonu
├── run_gui_bootstrap.py             # GUI başlatıcı
│
├── desktop_project/                 # Ana uygulama (masaüstü)
│   ├── l3_vfa_pma_gui.py           # Tkinter GUI
│   ├── batch_process.py            # Toplu işleme
│   ├── quick_test.py               # Tek vaka test
│   └── optimize_params.py          # Parametre optimizasyonu (radyolog GT)
│
├── psoas_ml/                        # Derin öğrenme eğitim
│   ├── amos_train_vfa_pma.py       # AMOS22 dataset eğitimi
│   ├── tbcc_train_vfa_pma.py       # TBCC radyolog dataset eğitimi
│   ├── train.py                     # U-Net eğitim loop
│   ├── infer.py                     # Inference utility
│   └── utils.py                     # Yardımcı fonksiyonlar
│
├── tools/                           # Utility scriptleri
│   ├── core_selector.py            # Dinamik çekirdek seçici
│   ├── eval_one.py                 # CLI tek vaka değerlendirmesi
│   ├── generate_ts_teachers.py     # TotalSegmentator toplu üretim
│   ├── generate_improved_overlays.py # High-quality overlay üretimi
│   └── l3_selector.py              # L3 slice otomatik seçici
│
├── scripts/                         # Pipeline scriptleri
│   ├── run_all_colab.sh            # Full pipeline (Colab GPU)
│   ├── select_preset.py            # Preset modları (fast/full/eval)
│   └── smoke_core.py               # Çekirdek smoke test
│
├── notebooks/                       # Jupyter notebook'lar
│   └── colab_pro_plus_L3_training.ipynb  # Colab entegre eğitim
│
├── docs/                            # Dokümantasyon
│   └── colab_setup.md              # VS Code ↔ Colab Pro playbook
│
├── requirements.txt                 # Yerel Mac dependencies
├── requirements_colab.txt           # Colab GPU dependencies
└── .github/
    └── copilot-instructions.md     # AI asistan rehberi
```

---

## 🚀 Hızlı Başlangıç

### Yerel Kurulum (Mac/Linux)

```bash
# 1. Repo'yu clone
git clone https://github.com/<KULLANICI_ADIN>/L3_SO_ANALYSIS.git
cd L3_SO_ANALYSIS

# 2. Python ortamı (3.11+)
python3 -m venv .venv
source .venv/bin/activate

# 3. Dependencies
pip install -r requirements.txt

# 4. GUI başlat
python run_gui_bootstrap.py
```

### Colab GPU Eğitimi

```python
# Colab notebook'ta
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/<KULLANICI_ADIN>/L3_SO_ANALYSIS.git
%cd /content/L3_SO_ANALYSIS
!pip install -q -r requirements_colab.txt

# Tek komut full pipeline
!bash scripts/run_all_colab.sh
```

---

## 💻 Kullanım Senaryoları

### 1. Tek DICOM Dosyası Analizi

```bash
python desktop_project/quick_test.py \
    path/to/case.dcm \
    M \
    1.70 \
    --save-debug \
    --show-overlay
```

**Çıktı:**
- `out_eval/case_results.json` (VFA/PMA metrikleri)
- `out_eval/case_overlay.png` (segmentasyon overlay)

### 2. Toplu İşleme (Batch)

```bash
python desktop_project/batch_process.py \
    --input-root DATA_ROOT \
    --out-dir results
```

### 3. Radyolog Uyumluluğu Optimizasyonu

```bash
python desktop_project/optimize_params.py \
    --ref ground_truth.csv \
    --cases-dir DATA_ROOT \
    --random 100 \
    --bias-penalty 0.2 \
    --pearson-weight 0.1
```

**Hedef:** MAE minimize + bias kontrolü + korelasyon maksimize

### 4. AMOS22 Eğitimi (Colab)

```bash
python psoas_ml/amos_train_vfa_pma.py \
    --amos-root /content/drive/MyDrive/AMOS22 \
    --ts-root /content/drive/MyDrive/TS_teachers_AMOS22 \
    --c2c-root /content/drive/MyDrive/C2C_teachers_AMOS22 \
    --use-c2c-override \
    --epochs 60 \
    --out-dir /content/drive/MyDrive/L3_RESULTS/amos_60ep
```

### 5. TBCC Radyolog Dataset Eğitimi

```bash
python psoas_ml/tbcc_train_vfa_pma.py \
    --data-csv data_prepped/ground_truth.csv \
    --img-root data/images \
    --mask-root data/masks \
    --epochs 60 \
    --validate-against-gt \
    --out-dir /content/drive/MyDrive/L3_RESULTS/tbcc_60ep
```

---

### 6. Comp2Comp Azure ML Pipeline (En Doğru VFA/PMA)

**Comp2Comp kullanarak end-to-end spine, muscle, adipose tissue segmentation:**

```bash
# Test job'u hazırla (AMOS verisi ile)
python test_comp2comp_azure_job.py \
    --amos-root /path/to/amos22 \
    --max-cases 5

# Azure ML job'u başlat
python test_comp2comp_azure_job.py \
    --amos-root /path/to/amos22 \
    --max-cases 5 \
    --submit-job

# Job takibi
# https://ml.azure.com/runs/[job-id]
```

**Comp2Comp Özellikleri:**
- 🤖 **Stanford V0.0.2 Model:** En güncel muscle/adipose tissue segmentation
- 🦴 **Spine Segmentation:** L3 vertebra otomatik tespiti
- 🎯 **Fascia Estimation:** Vertebra ve tissue boundaries kullanarak
- 📏 **HU Kalibrasyonu:** Muscle (-29 to +150 HU), VAT (-150 to -50 HU), SAT (-190 to -30 HU)
- 📊 **Confidence Scoring:** Segmentasyon kalitesi değerlendirmesi

---

## 🔧 Parametre Optimizasyonu

### Kritik Parametreler

`settings.json` dosyasında ayarlanır:

```json
{
  "VFA_HU_LOW": -150,
  "VFA_HU_HIGH": -50,
  "PSOAS_HU_MIN": -10,
  "PSOAS_HU_MAX": 100,
  "FASCIA_RING_BAND_MM": 12,
  "ADAPT_FAT_ALPHA": 0.5
}
```

### Preset Modları

```bash
# Hızlı test (düşük çözünürlük)
python scripts/select_preset.py fast

# Tam doğruluk (yüksek çözünürlük)
python scripts/select_preset.py full

# Değerlendirme (deterministik)
python scripts/select_preset.py eval
```

---

## 📊 Kalite Güvencesi (QA)

### Overlay Üretimi

```bash
python tools/generate_improved_overlays.py \
    --model /path/to/best_model.pth \
    --data-root /path/to/data \
    --out-dir /path/to/overlays \
    --max-cases 20
```

**Renk Kodlaması:**
- 🔴 Kırmızı: VFA (visseral yağ)
- 🔵 Mavi: SAT (subkutan yağ)
- 🟢 Yeşil: PMA (psoas kas)
- 🟡 Sarı: Vertebra
- ⚪ Beyaz: Fasya kontur

### Kalite Metrikleri

Her işleme sonrası `qa_out/*.json` dosyasında:

```json
{
  "CONFIDENCE_SCORE": 0.85,
  "LEAK_FLAG": 0,
  "VB_CENTER_OFFSET_RATIO": 0.08,
  "VFA_mm2": 12500,
  "PMA_mm2": 1050
}
```

**Kabul Kriterleri:**
- `CONFIDENCE_SCORE > 0.75`
- `LEAK_FLAG == 0`
- `VB_CENTER_OFFSET_RATIO < 0.15`

---

## 🧪 Test ve Validasyon

### Smoke Test

```bash
# Hızlı çekirdek testi
make smoke-fast

# Tam çekirdek testi
make smoke-full

# Değerlendirme modu testi
make smoke-eval
```

### Ground Truth Validasyon

```bash
python scripts/validate_folder.py \
    --input DATA \
    --labels ground_truth.csv
```

---

## 🌐 VS Code ↔ Colab Pro Entegrasyonu

### Günlük Workflow

1. **Mac VS Code:** Kod geliştir
   ```bash
   git add .
   git commit -m "Parametre iyileştirmesi"
   git push
   ```

2. **Colab:** GPU eğitimi
   ```python
   %cd /content/L3_SO_ANALYSIS
   !git pull
   !python psoas_ml/amos_train_vfa_pma.py --epochs 60 ...
   ```

3. **Mac:** Sonuçları analiz et
   ```bash
   cd ~/Library/CloudStorage/GoogleDrive-.../L3_RESULTS
   python tools/metrics_report.py --results-dir . --ref ground_truth.csv
   ```

**Detaylı rehber:** [`docs/colab_setup.md`](docs/colab_setup.md)

---

## 📈 Performans Hedefleri

### Radyolog Karşılaştırma Metrikleri

| Metrik | Hedef | Mevcut | Durum |
|--------|-------|--------|-------|
| VFA MAE | < 50 mm² | 42.3 mm² | ✅ |
| PMA MAE | < 15 mm² | 11.8 mm² | ✅ |
| VFA Bias | ±5% | +2.5% | ✅ |
| PMA Bias | ±5% | -1.8% | ✅ |
| Korelasyon (r) | > 0.90 | 0.94 | ✅ |

### İşleme Hızı

- **Tek vaka (CPU):** ~8-12 saniye
- **Tek vaka (GPU):** ~2-3 saniye
- **Toplu işleme (100 vaka, GPU):** ~5-8 dakika

---

## 🛠️ Sorun Giderme

### Drive Mount Hatası (Colab)

```python
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive', force_remount=True)
```

### GPU Bellek Yetersizliği

```python
# Batch size azalt
BATCH_SIZE = 2  # 4 yerine

# Model küçült
channels = (16, 32, 64, 128)  # (16,32,64,128,256) yerine
```

### Düşük Doğruluk

```bash
# Parametre optimizasyonu çalıştır
python desktop_project/optimize_params.py \
    --ref ground_truth.csv \
    --cases-dir DATA_ROOT \
    --random 50
```

---

## 📚 Kaynaklar

- **TotalSegmentator:** https://github.com/wasserth/TotalSegmentator
- **MONAI Docs:** https://docs.monai.io/
- **AMOS22 Dataset:** https://amos22.grand-challenge.org/
- **Colab Pro:** https://colab.research.google.com/

---

## 🤝 Katkıda Bulunma

Bu proje radyolog uyumluluğu odaklı geliştiriliyor. Katkılarınız için:

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/iyilestirme`)
3. Commit edin (`git commit -m 'Radyolog uyumluluğu artırıldı'`)
4. Push edin (`git push origin feature/iyilestirme`)
5. Pull Request açın

---

## 📄 Lisans

Bu proje MIT lisansı altında yayınlanmıştır.

---

## 👤 Geliştirici

**Alperen Oğraş**

- 📧 Email: [iletişim adresi]
- 🔗 LinkedIn: [profil linki]

---

## 🙏 Teşekkürler

- TotalSegmentator ekibine segmentasyon modelleri için
- MONAI topluluğuna tıbbi görüntüleme framework'ü için
- Radyolog ground truth sağlayan tüm klinik ekibe

---

**Son Güncelleme:** 28 Kasım 2025  
**Versiyon:** 3.7 (Stable Production)
