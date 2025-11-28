# VS CODE ↔ COLAB PRO TAM ENTEGRASYON PLAYBOOK

Bu dokümantasyon, Mac VS Code ortamında geliştirilen L3 VFA/PMA projesini Google Colab Pro GPU ile sorunsuz entegre etmek için adım adım rehber.

## 1. ÖN HAZIRLIK – GEREKENLER

### Hesaplar & Servisler
- Google hesabı + **Colab Pro/Pro+** aboneliği
- GitHub repo: `L3_SO_ANALYSIS` (mevcut)
- Google Drive: AMOS22, TS_teachers, C2C_teachers klasörleri

### Mac Ortamı
- VS Code kurulu
- Git yüklü
- Python 3.11 + venv (mevcut)
- rclone (Drive senkronizasyonu için)

### VS Code Eklentileri
Aşağıdaki eklentileri Extensions menüsünden yükle:

```
- Python (ms-python.python)
- Jupyter (ms-toolsai.jupyter)
- GitHub Pull Requests and Issues (isteğe bağlı)
- Colab for VS Code (veya benzeri Colab entegrasyonu)
```

---

## 2. REPO YAPISINI HAZIRLA

### Proje Klasör Yapısı
```
L3_SO_ANALYSIS/
├── docs/
│   └── colab_setup.md           # Bu dosya
├── psoas_ml/
│   ├── tbcc_train_vfa_pma.py    # TBCC dataset eğitim
│   ├── amos_train_vfa_pma.py    # AMOS22 dataset eğitim
│   └── infer.py                 # Inference script
├── tools/
│   ├── generate_improved_overlays.py
│   └── qa_amos_ts_c2c.py
├── notebooks/
│   └── colab_pro_plus_L3_training.ipynb
├── requirements_colab.txt       # Colab GPU dependencies
├── requirements.txt             # Yerel Mac dependencies
└── README.md
```

### `requirements_colab.txt` İçeriği
```txt
torch>=2.0.0
torchvision
torchaudio
monai>=1.2.0
nibabel
pydicom
opencv-python-headless
matplotlib
seaborn
tqdm
pandas
SimpleITK
scipy
scikit-image
totalsegmentator
ipywidgets
```

### GitHub'a Push
```bash
cd ~/Desktop/L3_SO_ANALYSIS
git add requirements_colab.txt docs/colab_setup.md
git commit -m "Colab Pro entegrasyon altyapısı eklendi"
git push origin main
```

---

## 3. COLAB'TE REPO'YU KULLAN

### İlk Kurulum (Her Yeni Colab Session'ında)

```python
# 1) Drive mount
from google.colab import drive
drive.mount('/content/drive')

# 2) Repo'yu clone (ilk kez) veya pull (güncel çek)
import os
if not os.path.exists('/content/L3_SO_ANALYSIS'):
    !git clone https://github.com/<KULLANICI_ADIN>/L3_SO_ANALYSIS.git
    %cd /content/L3_SO_ANALYSIS
else:
    %cd /content/L3_SO_ANALYSIS
    !git pull

# 3) Dependencies
!pip install -q -r requirements_colab.txt

# 4) GPU kontrolü
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

### Runtime Ayarları
- **Runtime → Change runtime type**
  - Hardware accelerator: **GPU** (T4 veya V100)
  - Runtime shape: **High-RAM** (Pro+ varsa)

---

## 4. VS CODE ↔ COLAB GÜNLÜK WORKFLOW

### Senaryo 1: VS Code'da Kod Geliştir, Colab'ta Eğit

#### Mac VS Code Tarafı
```bash
# 1) Kod yaz/düzenle
code ~/Desktop/L3_SO_ANALYSIS/psoas_ml/tbcc_train_vfa_pma.py

# 2) Değişiklikleri GitHub'a gönder
git add .
git commit -m "VFA threshold optimizasyonu - radyolog uyumluluğu artırıldı"
git push
```

#### Colab Tarafı
```python
# 3) Son değişiklikleri çek
%cd /content/L3_SO_ANALYSIS
!git pull

# 4) Eğitimi başlat
!python psoas_ml/tbcc_train_vfa_pma.py \
    --epochs 60 \
    --batch-size 4 \
    --lr 1e-3 \
    --save-dir /content/drive/MyDrive/L3_RESULTS/tbcc_run_001
```

### Senaryo 2: VS Code Colab Eklentisi ile Doğrudan

1. **VS Code → Command Palette** (`Cmd+Shift+P`)
2. **"Colab: Sign in"** → Google hesabınla giriş
3. **"Colab: Open Notebook"** → `colab_pro_plus_L3_training.ipynb`
4. Kod hücrelerini VS Code'da yaz, Colab GPU'da çalıştır
5. Çıktıları VS Code Output panelinde gör

---

## 5. EĞİTİM & SONUÇ YÖNETİMİ

### Eğitim Çıktılarını Drive'a Kaydet

```python
# Colab script içinde
OUT_DIR = Path('/content/drive/MyDrive/L3_RESULTS')
RUN_DIR = OUT_DIR / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Model checkpoints
torch.save(model.state_dict(), RUN_DIR / 'best_model.pth')

# QA overlays
shutil.copytree('/content/qa_out', RUN_DIR / 'qa_overlays')

# Metrics JSON
with open(RUN_DIR / 'metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Mac'te Sonuçları İncele

```bash
# Drive senkron edilmiş olduğunda
cd ~/Library/CloudStorage/GoogleDrive-<MAIL>/MyDrive/L3_RESULTS/run_20251128_143022

# Overlay'leri gör
open qa_overlays/

# Metrikleri analiz et
python tools/metrics_report.py --results-dir . --ref ground_truth.csv
```

---

## 6. PROJE ÖZELİ ÖNERİLER

### L3 VFA/PMA Senaryosu İçin En İyi Pratikler

#### 1. Veri Organizasyonu
```
Google Drive/
├── AMOS22/
│   └── imagesTr/*.nii.gz
├── TS_teachers_AMOS22/
│   └── <case_id>/
│       ├── vertebrae_lumbar.nii.gz
│       ├── psoas_major_left.nii.gz
│       └── ...
├── C2C_teachers_AMOS22/
│   ├── <case_id>_vat.npy
│   ├── <case_id>_sat.npy
│   └── <case_id>_psoas_left.npy
└── L3_RESULTS/
    └── run_YYYYMMDD_HHMMSS/
```

#### 2. İş Bölümü: Mac vs Colab

**Mac VS Code'da:**
- Kod geliştirme (.py dosyaları)
- GUI debug (hafif örnekler)
- Overlay kalite kontrolü (tek vaka)
- Ground truth karşılaştırma (lokal CSV)

**Colab GPU'da:**
- TotalSegmentator teacher üretimi (toplu)
- UNet eğitimi (60 epoch, GPU)
- Batch inference (tüm test seti)
- Parametre optimizasyonu (random search)

#### 3. Eğitim Scriptleri

**`psoas_ml/tbcc_train_vfa_pma.py`** için tipik çağrı:
```bash
python psoas_ml/tbcc_train_vfa_pma.py \
    --data-csv /content/drive/MyDrive/tbcc_ground_truth.csv \
    --img-root /content/drive/MyDrive/TBCC_images \
    --mask-root /content/drive/MyDrive/TBCC_masks \
    --epochs 60 \
    --batch-size 8 \
    --lr 1e-3 \
    --save-every 10 \
    --out-dir /content/drive/MyDrive/L3_RESULTS/tbcc_60ep
```

**`psoas_ml/amos_train_vfa_pma.py`** için tipik çağrı:
```bash
python psoas_ml/amos_train_vfa_pma.py \
    --amos-root /content/drive/MyDrive/AMOS22 \
    --ts-root /content/drive/MyDrive/TS_teachers_AMOS22 \
    --c2c-root /content/drive/MyDrive/C2C_teachers_AMOS22 \
    --epochs 60 \
    --use-c2c-override \
    --out-dir /content/drive/MyDrive/L3_RESULTS/amos_c2c_60ep
```

---

## 7. SORUN GİDERME

### Drive Mount Hatası
```python
# Eğer mount hata verirse:
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive', force_remount=True)
```

### Git Conflict
```bash
# Colab'ta çakışma olursa:
%cd /content/L3_SO_ANALYSIS
!git stash
!git pull
!git stash pop
```

### GPU Bellek Yetersizliği
```python
# Batch size azalt
BATCH_SIZE = 2  # 4 yerine

# Mixed precision kullan (zaten varsayılan)
USE_AMP = True

# Model küçült
channels = (16, 32, 64, 128)  # (16,32,64,128,256) yerine
```

### TotalSegmentator İndirme Hatası
```bash
# İlk çalışmada model ağırlıkları indirilir (~2GB)
# Timeout olursa:
!pip install --upgrade totalsegmentator
!python -c "from totalsegmentator.libs import download_pretrained_weights; download_pretrained_weights(1)"
```

---

## 8. KISA ÖZET – ADIM ADIM

### İlk Kurulum (Tek Sefer)
1. VS Code'da Python, Jupyter, Colab eklentilerini yükle
2. `requirements_colab.txt` oluştur ve GitHub'a push et
3. Colab'ta repo'yu clone et
4. Dependencies yükle

### Günlük Kullanım
1. **VS Code:** Kod değiştir → commit → push
2. **Colab:** git pull → eğitimi çalıştır
3. **Sonuçlar:** Drive'a kaydedilir → Mac'te analiz et

### Kritik Hatırlatmalar
- ✅ Her Colab session'ı sonunda **`drive.flush_and_unmount()`** çağır (veri kaybını önler)
- ✅ Uzun eğitimlerde **checkpoint kaydetmeyi** ihmal etme (Colab 12 saat sonra kesilir)
- ✅ Radyolog uyumluluğunu **her 10 epoch'ta** kontrol et (`optimize_params.py`)
- ✅ Overlay QA'yı **manuel gözden geçir** (otomatik metrikler yeterli değil)

---

## 9. İLERİ SEVİYE: TEK KOMUT FULL EĞİTİM

### `scripts/run_all_colab.sh` (İleride eklenecek)

```bash
#!/bin/bash
# Tüm teacher üretimi + eğitim + QA pipeline

python tools/generate_ts_teachers.py --amos-root $AMOS_ROOT --out-dir $TS_ROOT
python tools/generate_c2c_teachers.py --cases-dir $DATA_DIR --out-dir $C2C_ROOT
python psoas_ml/amos_train_vfa_pma.py --epochs 60 --out-dir $RESULTS_DIR
python tools/qa_amos_ts_c2c.py --model $RESULTS_DIR/best_model.pth --summary
```

Kullanım:
```bash
# Colab'ta
!bash scripts/run_all_colab.sh
```

---

## 10. KAYNAKLAR & BAĞLANTILAR

- **TotalSegmentator:** https://github.com/wasserth/TotalSegmentator
- **MONAI Docs:** https://docs.monai.io/
- **Colab Pro Limits:** https://research.google.com/colaboratory/faq.html
- **rclone Drive Sync:** https://rclone.org/drive/

---

**Son Güncelleme:** 28 Kasım 2025  
**Proje:** L3 VFA/PMA Radyolog Uyumluluğu Analizi  
**Geliştirici:** Alperen Oğraş
