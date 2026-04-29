# 📤 DICOMNET Verilerini Colab'a Yükleme Rehberi

Bu rehber DICOMNET lokal verilerinizi Google Drive üzerinden Colab'a taşımanızı sağlar.

## 🎯 Amaç

- DICOMNET klasöründeki ~45 vaka
- Google Drive'a yükleme
- Colab'da C2C teacher mask üretimi (GPU ile hızlı)

## 📊 Veri Boyutu Tahmini

```bash
# DICOMNET boyutunu kontrol edin
du -sh ~/Desktop/DICOMNET
```

Tipik boyut: **2-5 GB** (45 vaka için)

## 🚀 Yükleme Seçenekleri

### Seçenek 1: Google Drive Desktop App (ÖNERİLEN)

1. **Google Drive Desktop uygulamasını kurun**
   - [https://www.google.com/drive/download/](https://www.google.com/drive/download/)

2. **DICOMNET klasörünü Google Drive'a kopyalayın**
   ```bash
   # Finder'dan sürükle-bırak veya terminal:
   cp -r ~/Desktop/DICOMNET ~/Google\ Drive/My\ Drive/
   ```

3. **Senkronizasyon bekleyin**
   - Google Drive simgesine tıklayarak ilerlemeyi takip edin
   - 2-5 GB = yaklaşık 10-30 dakika (internet hızınıza göre)

### Seçenek 2: rclone (Komut Satırı)

```bash
# rclone kurulumu (macOS)
brew install rclone

# Google Drive yapılandırması
rclone config
# İsim: gdrive
# Storage: drive
# Tarayıcıda kimlik doğrulama

# DICOMNET'i yükle
rclone copy ~/Desktop/DICOMNET gdrive:DICOMNET -P --transfers 8
```

### Seçenek 3: Web Arayüzü (Manuel)

1. [https://drive.google.com](https://drive.google.com) açın
2. "Yeni" → "Klasör yükle"
3. DICOMNET klasörünü seçin
4. Yükleme tamamlanana kadar bekleyin

**Not:** Büyük veri setleri için yavaş olabilir.

### Seçenek 4: Google Colab Upload (Küçük Test İçin)

```python
# Colab notebook içinde
from google.colab import files
import zipfile
import os

# ZIP yükle
uploaded = files.upload()  # DICOMNET.zip seçin

# Aç
for fn in uploaded.keys():
    with zipfile.ZipFile(fn, 'r') as zip_ref:
        zip_ref.extractall('/content/')
```

**Not:** 2GB+ dosyalar için önerilmez (timeout riski).

## ✅ Doğrulama

Google Drive'da veri kontrolü:

```python
# Colab'da çalıştırın
from google.colab import drive
drive.mount('/content/drive')

import os
dicom_path = '/content/drive/MyDrive/DICOMNET'
cases = [d for d in os.listdir(dicom_path) if os.path.isdir(os.path.join(dicom_path, d))]
print(f"✅ {len(cases)} vaka bulundu")

# İlk 5 vaka
for case in cases[:5]:
    case_path = os.path.join(dicom_path, case)
    dcm_files = len([f for f in os.listdir(case_path) if f.endswith('.dcm')])
    print(f"  {case}: {dcm_files} DICOM")
```

Beklenen çıktı:
```
✅ 45 vaka bulundu
  102.000000-CT COLONOGRAPHY-.9865: 543 DICOM
  2.000000-Body 3.0 CE-77940: 148 DICOM
  ...
```

## 🔬 C2C Teacher Üretimi

Veriler Google Drive'a yüklendikten sonra:

1. **Colab notebook'u açın**
   - `notebooks/C2C_teachers_DICOMNET.ipynb`
   - Colab'a yükleyin

2. **GPU runtime seçin**
   - Runtime → Change runtime type → T4 GPU

3. **DICOMNET yolunu ayarlayın**
   ```python
   DICOMNET_PATH = '/content/drive/MyDrive/DICOMNET'
   ```

4. **Notebook'u çalıştırın**
   - Cell → Run all
   - ~2-3 saat sürer (45 vaka için)

## 📥 Sonuçları İndirme

Teacher maskler oluşturulduktan sonra:

```bash
# Google Drive Desktop ile (otomatik senkronize)
ls ~/Google\ Drive/My\ Drive/C2C_teachers_local/

# veya rclone ile
rclone copy gdrive:C2C_teachers_local ~/Desktop/C2C_teachers_local -P
```

## 🎓 İpuçları

### Hız Optimizasyonu

- **Colab Pro+** kullanın (A100 GPU ile 3-4x hızlı)
- **İnternet hızı** önemli (yükleme için)
- **Birden fazla oturum** açıp paralel işleyin (riski: kaynak limitleri)

### Veri Güvenliği

- DICOMNET'te **hasta bilgisi** varsa anonim hale getirin
- Google Drive **güvenlik ayarlarını** kontrol edin
- İşlem sonrası **Colab çalışma zamanını temizleyin**

### Sorun Giderme

**Problem:** "No space left on device"
```python
# Colab disk alanı kontrol
!df -h
# Gereksiz dosyaları sil
!rm -rf /content/sample_data
```

**Problem:** "Timeout during upload"
- Daha küçük gruplar halinde yükleyin (10 vaka × 5 batch)
- rclone `--transfers 4` parametresiyle deneyin

**Problem:** "C2C inference failed"
- GPU runtime'ın aktif olduğunu kontrol edin
- RAM kullanımını izleyin (Runtime → Manage sessions)

## 📞 Destek

Sorun yaşarsanız:
1. Colab log'larını kontrol edin
2. C2C manifest.json dosyasındaki hata nedenlerini inceleyin
3. GitHub Issues açın: [StanfordMIMI/Comp2Comp](https://github.com/StanfordMIMI/Comp2Comp/issues)
