# GitHub Codespaces ile L3 SO Analysis Kullanımı

## GitHub Codespaces Nedir?
GitHub Codespaces, Azure altyapısında çalışan bulut tabanlı bir VS Code geliştirme ortamıdır. GPU desteği ile Colab benzeri deneyimi VS Code içinden tam kontrol ile sunar.

## Avantajlar
- ✅ VS Code'dan direkt hücre çalıştırma/debug
- ✅ GPU desteği (NVIDIA T4/A10G/A100)
- ✅ Repo ile entegre (değişiklikler direkt commit/push)
- ✅ Persistent storage (checkpointler kaybolmaz)
- ✅ SSH/port forwarding (Jupyter, TensorBoard)
- ✅ Colab'dan daha uzun süre limiti

## Hızlı Başlangıç

### 1. Codespace Oluştur (GPU ile)

**GitHub Web'den:**
1. https://github.com/alpogras23/L3-SO- → Code (yeşil buton)
2. Codespaces sekmesi → "Create codespace on psoas-improvement"
3. Machine type: **4-core, 16GB RAM, GPU** seç (GitHub Team/Enterprise için)
4. Create

**VS Code'dan:**
1. Komut Paleti (Cmd+Shift+P): `Codespaces: Create New Codespace`
2. Repo: `alpogras23/L3-SO-`
3. Branch: `psoas-improvement`
4. Machine type: GPU seç
5. Create

### 2. Otomatik Kurulum Bekle
- `.devcontainer/setup.sh` otomatik çalışır (5-10 dakika)
- CUDA, PyTorch, MONAI, TotalSegmentator, nnUNetv2 kurulur
- GPU kontrol çıktısını terminal'de göreceksin

### 3. Veri Yükleme

**Seçenek A: GitHub Secrets ile (küçük veri):**
```bash
# Codespace terminalinde
echo "$AMOS22_DATA" | base64 -d > amos22.tar.gz
tar -xzf amos22.tar.gz
```

**Seçenek B: Azure Blob/S3 (büyük veri, önerilir):**
```bash
az storage blob download-batch --account-name <storage> --source amos22 --destination ~/amos22_data
```

**Seçenek C: `gh` CLI ile artifact upload:**
```bash
# Lokal (Mac) terminalinde
gh codespace cp ~/Desktop/amos22 remote:/workspaces/L3-SO-/amos22_data -r
```

### 4. Notebook'u Aç ve Çalıştır
1. `notebooks/AMOS22_L3_TS_C2C_VFA_PMA_tek_tik.ipynb` dosyasını aç
2. Kernel seç: Python 3.10 (Codespace default env)
3. Yol değişkenlerini Codespace path'lerine ayarla:
   ```python
   AMOS22_ROOT = "/workspaces/L3-SO-/amos22_data"
   DICOM_OUT = "/workspaces/L3-SO-/outputs/dicom"
   TEACHERS_OUT = "/workspaces/L3-SO-/outputs/teachers"
   QA_OUT = "/workspaces/L3-SO-/qa_out"
   CKPT_DIR = "/workspaces/L3-SO-/ckpts"
   ```
4. Hücreleri sırayla çalıştır (Shift+Enter)

### 5. Çıktıları Kaydet/İndir
```bash
# Codespace'ten lokal'e kopyala
gh codespace cp remote:/workspaces/L3-SO-/qa_out ~/Desktop/L3_qa_results -r
```

## GPU Kotası ve Maliyetler

**GitHub Free:** Codespaces var ama GPU yok  
**GitHub Pro:** Aylık 120 saat (CPU), GPU ekstra ücretli  
**GitHub Team/Enterprise:** GPU destekli makineler mevcut

GPU maliyet: ~$0.18/saat (T4), ~$0.45/saat (A10G), ~$1.25/saat (A100)

## Sorun Giderme

**GPU görünmüyor:**
```bash
nvidia-smi
# Hata veriyorsa: Codespace makine tipini GPU'lu yeniden oluştur
```

**Kurulum hatası:**
```bash
# Manual kurulum
bash .devcontainer/setup.sh
```

**Süre sınırı (idle timeout):**
- Codespace'i durdur/başlat: VS Code sol alt > Codespace adı > Stop/Restart

## Alternatif: Lokal Dev Container

Eğer kendi GPU'lu Linux makineniz varsa:
```bash
# Docker + NVIDIA Container Toolkit kurulu olmalı
cd ~/Desktop/L3_SO_ANALYSIS
code .
# VS Code: "Reopen in Container"
```

---

## Özet Karşılaştırma

| Özellik | Colab | Codespaces (GPU) | Lokal |
|---------|-------|------------------|-------|
| VS Code Kontrol | ❌ | ✅ | ✅ |
| GPU | ✅ (ücretsiz sınırlı) | ✅ (ücretli) | ✅ (donanım) |
| Süre Limiti | 12h idle | Ayarlanabilir | ∞ |
| Persistent Storage | Drive | Repo/volume | ∞ |
| Hücre Debug | Sınırlı | Tam | Tam |
| Maliyet | Ücretsiz (Pro+) | ~$0.18-1.25/h | Sadece elektrik |

**Öneri:** AMOS22 eğitimi için Codespaces GPU (60 epok ~6-8 saat = $1-10).
