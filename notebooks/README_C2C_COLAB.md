# 🔬 C2C Teacher Mask Üretimi - Colab Rehberi

Bu klasörde DICOMNET lokal verilerinizden Comp2Comp kullanarak teacher mask üretimi için gerekli tüm dosyalar var.

## 📂 Dosyalar

- **`C2C_teachers_DICOMNET.ipynb`**: Colab notebook (GPU ile çalışır)
- **`../docs/DICOMNET_TO_COLAB_GUIDE.md`**: Detaylı yükleme ve kullanım rehberi

## ⚡ Hızlı Başlangıç

### 1️⃣ DICOMNET'i Google Drive'a Yükle

```bash
# Google Drive Desktop ile (ÖNERİLEN)
cp -r ~/Desktop/DICOMNET ~/Google\ Drive/My\ Drive/

# veya rclone ile
rclone copy ~/Desktop/DICOMNET gdrive:DICOMNET -P
```

**Veri boyutu:** ~2-5 GB (45 vaka)  
**Yükleme süresi:** 10-30 dakika

### 2️⃣ Colab Notebook'u Çalıştır

1. `C2C_teachers_DICOMNET.ipynb` dosyasını [Google Colab'da aç](https://colab.research.google.com/)
2. **Runtime → Change runtime type → T4 GPU** seç
3. **DICOMNET_PATH** değişkenini ayarla:
   ```python
   DICOMNET_PATH = '/content/drive/MyDrive/DICOMNET'
   ```
4. **Cell → Run all** ile başlat

**İşlem süresi:** 
- T4 GPU: ~2-3 saat (45 vaka)
- A100 GPU (Pro+): ~45-60 dakika

### 3️⃣ Sonuçları İndir

```bash
# Google Drive Desktop (otomatik senkronize)
ls ~/Google\ Drive/My\ Drive/C2C_teachers_local/

# veya rclone
rclone copy gdrive:C2C_teachers_local ~/Desktop/C2C_teachers_local -P
```

**Çıktı yapısı:**
```
C2C_teachers_local/
├── c2c_manifest.json (özet)
└── <case_id>/
    └── <timestamp>/
        └── <case_id>/
            └── *.nii.gz (VAT, SAT, psoas maskler)
```

## 🎯 Beklenen Sonuç

- ✅ **45 vaka için teacher maskler** (VAT, SAT, psoas)
- ✅ **Manifest JSON** (başarı/hata istatistikleri)
- ✅ **NIfTI format** (.nii.gz)

## 🐛 Sorun Giderme

### "No space left on device"
```python
!df -h  # Disk kontrolü
!rm -rf /content/sample_data  # Gereksiz dosyaları sil
```

### "C2C inference failed"
- GPU runtime aktif mi kontrol et
- RAM kullanımını izle (Runtime → Manage sessions)
- Manifest'teki error reason'a bak

### Yavaş yükleme
- Google Drive Desktop yerine rclone kullan
- 10'ar vaka gruplar halinde yükle
- İnternet bağlantısını kontrol et

## 📊 Performans

| GPU | Vaka/Dakika | 45 Vaka Toplam |
|-----|-------------|----------------|
| T4 (Colab Free) | ~0.4 | 2-3 saat |
| T4 (Colab Pro) | ~0.4 | 2-3 saat |
| A100 (Colab Pro+) | ~1.0 | 45-60 dk |

## 🔄 Sonraki Adımlar

1. ✅ Teacher maskler üretildi
2. 📥 Lokal makineye indir
3. 🎓 AMOS22 eğitim notebook'a entegre et
4. 🚀 Multi-teacher training başlat

## 📚 Referanslar

- [Comp2Comp GitHub](https://github.com/StanfordMIMI/Comp2Comp)
- [DICOMNET Yükleme Rehberi](../docs/DICOMNET_TO_COLAB_GUIDE.md)
- [Colab GPU Dökümanı](https://colab.research.google.com/notebooks/gpu.ipynb)

## 💡 İpuçları

- **Gece çalıştırın:** Colab Pro+ ile gece boyunca çalışabilir
- **Paralel işlem:** Farklı Colab oturumlarında vaka gruplarını paralel işleyin (dikkat: kaynak limitleri)
- **Veri güvenliği:** HIPAA uyumlu olmayan veriler için anonim hale getirin

---

**Sorularınız için:** [GitHub Issues](https://github.com/alpogras23/L3-SO-/issues)
