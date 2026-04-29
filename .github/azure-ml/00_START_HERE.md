
# 🚀 Azure ML L3 VFA/PMA Pipeline - BAŞLANGICI

**Özet:** Bu klasörde masaüstündeki AMOS22 verilerinizi Azure ML GPU üzerinde işlemek için gereken tüm dosyalar ve araçlar vardır.

---

## ⚡ 30 Saniye Başlangıç

```bash
cd /Users/alperenogras/Desktop/L3_SO_ANALYSIS/.github/azure-ml

# 1. Çevre değişkenlerini ayarla
export AZURE_SUBSCRIPTION_ID="your-sub-id"
export AZURE_RESOURCE_GROUP="l3-rg"
export AZURE_ML_WORKSPACE="l3-ws"

# 2. Setup yapıp workspace oluştur
chmod +x setup_azure_ml.sh
./setup_azure_ml.sh

# 3. AMOS22 datasını yükle
az storage blob upload-batch \
  --account-name $(jq -r '.storage_account' azure_ml_config.json) \
  --destination amos22-data \
  --source /path/to/AMOS22/

# 4. Pipeline başlat (rule-based VFA/PMA)
python launcher.py \
  --input_data_path azureml://datastores/workspaceblobstore/paths/amos22-data/ \
  --run_vfa_pma \
  --wait

# 5. Sonuçları indir
python config_helper.py download <RUN_ID>
```

---

## 📂 Dosyalar ve Kullanımları

### 🐍 Python Scripts

| Dosya | Amaç | Örnek |
|-------|------|-------|
| **launcher.py** | Ana entry point - job'ları başlatır | `python launcher.py --run_vfa_pma --wait` |
| **run_l3_vfa_pma.py** | Batch processing (L3, VFA, PMA) | (Azure ML tarafından otomatik çalıştırılır) |
| **train_unet_azureml.py** | U-Net eğitimi | (Azure ML tarafından otomatik çalıştırılır) |
| **config_helper.py** | Workspace kontrol + run izleme | `python config_helper.py status` |
| **generate_report.py** | Rapor oluşturma | (Azure ML step'i tarafından) |
| **prepare_training_data.py** | Training data hazırlığı | (Azure ML step'i tarafından) |

### 🔧 Bash Scripts

| Dosya | Amaç | Çalıştırma |
|-------|------|-----------|
| **setup_azure_ml.sh** | Workspace + compute + storage kurma | `./setup_azure_ml.sh` |
| **quick_setup.sh** | Hızlı test setup'ı | `./quick_setup.sh` |
| **START_HERE.sh** | Etkileşimli başlangıç | `./START_HERE.sh` |

### 📋 Konfigürasyon

| Dosya | İçerik |
|-------|--------|
| **environment.yml** | Python bağımlılıkları (pip packages) |
| **pipeline.yml** | Azure ML pipeline tanımı |
| **training-job.yml** | U-Net eğitim job tanımı |

### 📚 Dokümantasyon

| Dosya | Kapsamı | Oku |
|-------|---------|-----|
| **QUICKSTART.md** | 7 adımlı başlangıç + örnekler | ⭐ OKUMALI |
| **ARCHITECTURE.md** | Pipeline tasarım + mimarisi | Pipeline hakkında bilgi |
| **README.md** | Proje özeti | Genel bakış |
| **README_PIPELINE.md** | Pipeline detayları | Deep dive |
| **DEPLOYMENT_CHECKLIST.md** | Production deployment | Canlıya çıkış için |

---

## 🎯 İş Akışı (Workflow)

```
┌─────────────────┐
│  START_HERE.sh  │  ← Buradan başla (etkileşimli)
└────────┬────────┘
         ↓
┌──────────────────────┐
│ setup_azure_ml.sh    │  ← Workspace oluştur
└────────┬─────────────┘
         ↓
┌──────────────────────┐
│ AMOS22 datasını      │  ← Azure Storage'a yükle
│ Storage'a yükle      │
└────────┬─────────────┘
         ↓
┌──────────────────────┐
│ launcher.py          │  ← Job'ları başlat
│ --run_vfa_pma        │
└────────┬─────────────┘
         ↓
┌──────────────────────┐
│ GPU Processing       │  ← Azure ML'de çalışır
│ (L3, VFA, PMA)       │
└────────┬─────────────┘
         ↓
┌──────────────────────┐
│ Sonuçları indir      │  ← config_helper.py download
│ ve analiz et         │
└──────────────────────┘
```

---

## ✨ Yaygın Görevler

### Task 1: Basit VFA/PMA Hesaplama (30 dakika)

```bash
# Setup
./setup_azure_ml.sh

# AMOS22 50 vakalık subset'i yükle
az storage blob upload-batch \
  --account-name <STORAGE> \
  --destination amos22-test \
  --source ~/Desktop/AMOS22/subset50/

# Process
python launcher.py \
  --input_data_path azureml://datastores/.../amos22-test/ \
  --run_vfa_pma \
  --wait

# Download
python config_helper.py download <RUN_ID> --output ./results_test
```

---

### Task 2: U-Net Eğitimi (4-6 saat)

```bash
# Full dataset'i yükle (500 vaka)
az storage blob upload-batch \
  --account-name <STORAGE> \
  --destination amos22-full \
  --source ~/Desktop/AMOS22/

# Eğitimi başlat
python launcher.py \
  --input_data_path azureml://datastores/.../amos22-full/ \
  --run_training \
  --wait

# Model indir
python config_helper.py download <TRAINING_RUN_ID> --output ./models
```

---

### Task 3: Hybrid Mode (Rule-based + DL) (3-4 saat)

```bash
# Önceki eğitimden model var mı kontrol et
ls models/outputs/best_unet.pt

# Hybrid processing başlat
python launcher.py \
  --input_data_path azureml://datastores/.../amos22-data/ \
  --run_vfa_pma \
  --use_dl \
  --model_path ./models/best_unet.pt \
  --wait
```

---

## 🔍 Debugging & Monitoring

### Run durumunu kontrol et

```bash
# Son run'ları göster
python config_helper.py list_runs --limit 5

# Spesifik run'ı izle
python config_helper.py monitor <RUN_ID>

# Workspace status
python config_helper.py status
```

### Log'ları indir

```bash
# Run çıktılarını indir
az ml run download --run-id <RUN_ID> --output ./logs_<RUN_ID>

# Hata mesajlarını ara
grep -r "error\|Error\|ERROR" logs_<RUN_ID>/
```

---

## 💡 İpuçları

### 1. İlk Kez Çalıştırmadan Önce

```bash
# Azure CLI yükleme kontrolü
az --version

# Python version kontrolü
python --version  # 3.9+ gerekli

# Subscription kontrol
az account show
```

### 2. Maliyet Kontrolü

```bash
# GPU maliyeti kontrol et (T4 ~ $0.35/saat)
# Storage işlemleri: blob upload batch ~free, 
# storage: $0.024/GB month

# Bir çalıştırma = ~$0.50 (50 vaka, 1 saat)
```

### 3. Hızlı Test

```bash
# 5 vakalık test datasıyla başla
az storage blob upload-batch \
  --account-name <STORAGE> \
  --destination amos22-tiny \
  --source ~/Desktop/AMOS22/ \
  --pattern "amos_000[0-5]*"

# Hızlı process
python launcher.py \
  --input_data_path azureml://datastores/.../amos22-tiny/ \
  --run_vfa_pma \
  --wait
```

---

## 🆘 Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| "Workspace not found" | `az ml folder attach --workspace-name <WS> --resource-group <RG>` |
| "Storage key not found" | `az storage account keys list --account-name <STORAGE> --resource-group <RG>` |
| "Module not found" | `pip install -r requirements.txt` |
| "GPU timeout" | Job'ı 8 saat timeout ile yeniden çalıştır |
| "Low memory" | batch_size'ı 8→4 düşür |

---

## 📞 İletişim

- **Issues:** GitHub'da issue açın
- **Questions:** Dokümantasyonu kontrol et (QUICKSTART.md)
- **Feedback:** GitHub discussions

---

## 🏁 Sonraki Adımlar

1. **QUICKSTART.md'yi oku** - Detaylı 7-adımlı rehber
2. **setup_azure_ml.sh'i çalıştır** - Workspace oluştur
3. **AMOS22 datasını yükle** - Azure Storage'a
4. **launcher.py'i çalıştır** - İlk job'ı başlat
5. **Sonuçları analiz et** - Radyolog validasyonu

---

✨ **Azure ML L3 VFA/PMA Pipeline Başarıyla Kuruldu!** ✨

**Şimdi QUICKSTART.md dosyasını okuyarak devam etmeye başla!**

---

**Tarih:** Aralık 2025  
**Version:** 1.0  
**Status:** Ready to Use ✅
