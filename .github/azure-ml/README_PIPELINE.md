# Azure ML L3 VFA/PMA Analysis Pipeline

Masaüstündeki AMOS22 verilerini kullanarak L3 seviyesinde VFA/PMA hesaplamalarını GPU üzerinde otomatikleştiren Azure ML pipeline'ı.

## 📋 Genel Bakış

### Pipeline Adımları

1. **Rule-based VFA/PMA Processing** - L3 slice'ında anatomik parameterler ile VFA/PMA hesapla
2. **Training Data Preparation** - DL model eğitimi için 3-channel input/output hazırla
3. **U-Net Model Training** - MONAI U-Net'i 60 epoch boyunca eğit (GPU)
4. **Hybrid VFA/PMA Processing** - Eğitilmiş model + rule-based blend ile re-process
5. **Report Generation** - İki metodun karşılaştırması ve özet rapor

### Çıktılar

- `vfa_pma_results/` - Rule-based sonuçları (JSON + PNG overlays)
- `training_data/` - DL eğitimi için hazırlanan veri
- `trained_model/` - best_unet.pt checkpoint
- `hybrid_results/` - DL-enhanced sonuçları
- `final_report.json` - Karşılaştırmalı analiz raporu
- `comparison_plots.png` - Görsel karşılaştırmalar

---

## 🚀 Hızlı Başlangıç

### 1. Azure CLI Kurulumu ve Kaynakları Oluştur

```bash
cd /Users/alperenogras/Desktop/L3_SO_ANALYSIS/.github/azure-ml/

# Setup script'i çalıştır (parametreleri düzenle)
chmod +x setup_azure_ml.sh
./setup_azure_ml.sh \
  <SUBSCRIPTION_ID> \
  <RESOURCE_GROUP> \
  eastus \
  l3-vfa-pma-workspace \
  l3vfapmastorage \
  l3-gpu-cluster \
  Standard_NC4as_T4_v3 \
  /Users/alperenogras/Desktop/L3_SO_ANALYSIS/data/AMOS22/ \
  /Users/alperenogras/Desktop/L3_SO_ANALYSIS/data/ts_masks/
```

**Parametreler:**
- `SUBSCRIPTION_ID`: Azure subscription ID'niz
- `RESOURCE_GROUP`: Mevcut Azure resource group'u
- `AMOS22 Path`: Masaüstündeki AMOS22 NIfTI dosyalarının yolu
- `TS Path`: TotalSegmentator maskelerinin yolu

### 2. Workspace Config Dosyası Oluştur

```bash
# Azure ML workspace config'ini indir
az ml folder attach \
  --resource-group <RESOURCE_GROUP> \
  --workspace-name l3-vfa-pma-workspace
```

Bu komut `.azureml/config.json` dosyasını oluşturacaktır.

### 3. Pipeline'ı Başlat

```bash
python launcher.py \
  --subscription_id <SUBSCRIPTION_ID> \
  --resource_group <RESOURCE_GROUP> \
  --workspace_name l3-vfa-pma-workspace \
  --compute_name l3-gpu-cluster \
  --scripts_dir ./ \
  --experiment_name l3_vfa_pma_analysis
```

Pipeline başladığında, run ID'niz ekranda gösterilecek. Azure Portal'da izlemeye devam edebilirsiniz.

### 4. Sonuçları İndir

```bash
# Run'ın tamamlanmasını bekle (Azure Portal'da izle)
# Sonrası:

az ml run download \
  --workspace-name l3-vfa-pma-workspace \
  --resource-group <RESOURCE_GROUP> \
  --run-id <RUN_ID> \
  --output ./results
```

---

## 📊 Dosya Yapısı

```
.github/azure-ml/
├── environment.yml                 # Conda dependencies
├── run_l3_vfa_pma.py              # Rule-based + DL VFA/PMA processor
├── prepare_training_data.py        # Training dataset hazırlayıcı
├── train_unet_model.py             # U-Net eğitim scripti
├── generate_report.py              # Analiz raporu üreticisi
├── launcher.py                     # Azure ML pipeline launcher
├── setup_azure_ml.sh               # Azure kaynakları setup scripti
└── README.md                        # Bu dosya
```

---

## 🔧 Özelleştirme

### Model Parametreleri

`train_unet_model.py` içinde değiştir:
- `--epochs`: 60 (default)
- `--batch_size`: 8 (default)
- `--learning_rate`: 1e-4 (default)

### Blend Factor

`run_l3_vfa_pma.py` içinde DL blending'i kontrol et:
- `--blend_alpha 0.0`: Pure rule-based
- `--blend_alpha 0.5`: 50/50 hybrid
- `--blend_alpha 1.0`: Pure DL

### Max Cases

`prepare_training_data.py` içinde:
```python
--max_cases 50  # İlk 50 vaka
```

Tüm dataset için:
```python
--max_cases -1  # Veya parametreyi kaldır
```

---

## 📈 Sonuçları Analiz Etme

Pipeline tamamlandıktan sonra:

```bash
# JSON sonuçlarını görüntüle
ls results/final_report.json
cat results/final_report.json | jq .

# Karşılaştırmalı grafikleri gör
open results/comparison_plots.png

# İndividual case sonuçları
ls results/vfa_pma_results/*_results.json | head -5
```

---

## 🐛 Sorun Giderme

### "Dataset not found" hatası

```bash
# Dataset'leri manuel olarak kaydet
python -c "
from azureml.core import Workspace
ws = Workspace.from_config()
from azureml.core import Dataset

# AMOS22
dataset = Dataset.File.from_files((ws.get_default_datastore(), 'amos22-data/'))
dataset.register(ws, name='amos22_dataset', create_new_version=True)

# TS Masks
dataset = Dataset.File.from_files((ws.get_default_datastore(), 'ts-masks/'))
dataset.register(ws, name='ts_masks_dataset', create_new_version=True)
"
```

### GPU compute "not found"

```bash
# Compute cluster status'ü kontrol et
az ml compute show \
  --workspace-name l3-vfa-pma-workspace \
  --resource-group <RESOURCE_GROUP> \
  --name l3-gpu-cluster
```

### Storage upload hatası

```bash
# Storage account key'lerini kontrol et
az storage account keys list \
  --account-name l3vfapmastorage \
  --resource-group <RESOURCE_GROUP>
```

---

## 📚 Bağlantılar

- [Azure ML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [MONAI U-Net](https://docs.monai.io/en/latest/networks.html#unet)
- [Azure CLI Reference](https://docs.microsoft.com/en-us/cli/azure/reference-index)

---

## ⚡ Performance Tahmini

| Step | Duration | GPU |
|------|----------|-----|
| VFA/PMA Rule-based | 30-60 min | 1x NC4as_T4_v3 |
| Data Preparation | 10-20 min | 1x NC4as_T4_v3 |
| U-Net Training (60 ep) | 30-45 min | 1x NC4as_T4_v3 |
| Hybrid VFA/PMA | 30-60 min | 1x NC4as_T4_v3 |
| Report Generation | 5-10 min | 1x NC4as_T4_v3 |
| **TOTAL** | **~2-3 hours** | |

---

## 📝 Notlar

- Pipeline adımları sırada çalışır (sequential)
- Hata durumunda tüm pipeline durdurulur
- Sonuçlar Azure Storage'da saklı kalır (sonra masaüstüne indirilir)
- GPU compute saat başına ücretlendirilir
- Min instances = 0, otomatik scale-down = 600 saniye (config edebilir)

---

Sorular? GitHub Issues'u açabilirsin! 🎯
