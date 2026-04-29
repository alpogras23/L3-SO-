# 🚀 Azure ML L3 VFA/PMA Pipeline - Hızlı Başlangıç

Bu rehber, AMOS22 verilerinizi Azure ML'de GPU üzerinde işlemek için adım adım talimatlar içerir.

---

## 📋 Ön Gereksinimler

- ✅ Azure subscription (ücretli veya ücretsiz deneme)
- ✅ Azure CLI kurulu (`az --version`)
- ✅ Python 3.9+
- ✅ AMOS22 NIfTI dosyaları (masaüstünüzde veya Azure Storage'da)
- ✅ TotalSegmentator maskelerine erişim

---

## 🔧 1. Azure ML Kaynakları Kurma

### Adım 1.1: Ortam Değişkenlerini Ayarla

```bash
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="l3-so-analysis-rg"
export AZURE_ML_WORKSPACE="l3-vfa-pma-ws"
export AZURE_REGION="eastus"
export AZURE_STORAGE_ACCOUNT="l3sofastg"  # Benzersiz isim
```

**Not:** Subscription ID'nizi almak için:
```bash
az account list --query "[].id" -o table
```

### Adım 1.2: Setup Script'ini Çalıştır

```bash
cd /Users/alperenogras/Desktop/L3_SO_ANALYSIS/.github/azure-ml
chmod +x setup_azure_ml.sh
./setup_azure_ml.sh
```

**Çıktı:**
```
✅ Resource Group hazır
✅ Storage Account hazır
✅ Azure ML Workspace hazır
✅ GPU Cluster hazır
```

Oluşturulan dosya: `azure_ml_config.json`

---

## 📦 2. AMOS22 Verilerini Azure Storage'a Yükle

### Adım 2.1: Masaüstünden Yükle

```bash
# Eğer AMOS22 veriler masaüstünüzde ise:
AMOS22_PATH="/Users/alperenogras/Desktop/AMOS22"
STORAGE_ACCOUNT=$(jq -r '.storage_account' azure_ml_config.json)

az storage blob upload-batch \
  --account-name "$STORAGE_ACCOUNT" \
  --destination amos22-data \
  --source "$AMOS22_PATH" \
  --pattern "*.nii.gz"
```

### Adım 2.2: İlerlemeyi Kontrol Et

```bash
az storage blob list \
  --account-name "$STORAGE_ACCOUNT" \
  --container-name amos22-data \
  --query "[length(@)]" -o tsv
```

---

## 🚀 3. Pipeline'ı Başlat

### Adım 3.1: VFA/PMA Processing (Rule-based)

```bash
python launcher.py \
  --input_data_path "azureml://datastores/workspaceblobstore/paths/amos22-data/" \
  --run_vfa_pma \
  --wait \
  --download \
  --download_dir "./results"
```

**Ne olur:**
1. ✅ AMOS22 vakalarını işle
2. ✅ L3 slice'ını tespit et
3. ✅ VFA/PMA hesapla
4. ✅ Overlay PNG'leri oluştur
5. ✅ Sonuçları JSON olarak kaydet

**Süre:** ~10-20 dakika (50 vaka için)

### Adım 3.2 (Opsiyonel): U-Net Eğitimi

Daha doğru tahminler için DL modeli eğit:

```bash
python launcher.py \
  --input_data_path "azureml://datastores/workspaceblobstore/paths/amos22-data/" \
  --run_training \
  --wait \
  --download \
  --download_dir "./models"
```

**Ne olur:**
1. ✅ Training data hazırla (50-500 vaka)
2. ✅ U-Net modelini eğit (60 epoch)
3. ✅ Best checkpoint'i kaydet
4. ✅ Training history CSV kaydet

**Süre:** ~2-4 saat (100 vaka, T4 GPU)

---

## 📊 4. Sonuçları İndir ve Analiz Et

### Adım 4.1: Sonuçları Kontrol Et

```bash
# VFA/PMA sonuçları
ls -lh results/outputs/
cat results/outputs/summary.json

# Eğitim sonuçları (varsa)
ls -lh models/outputs/
cat models/outputs/training_summary.json
```

### Adım 4.2: Spesifik Vaka Analizi

```bash
# Bir vakanın sonuçlarını göster
CASE_ID="amos_0001"
python << 'EOF'
import json
from pathlib import Path

results_file = Path("results/outputs") / f"{CASE_ID}_results.json"
with open(results_file) as f:
    data = json.load(f)

print(f"Case: {data['case_id']}")
print(f"VFA: {data['VFA_cm2']:.1f} cm²")
print(f"PMA: {data['PMA_cm2']:.1f} cm²")
print(f"Ratio: {data.get('VFA_PMA_ratio', 'N/A'):.2f}")
print(f"Method: {data['method']}")
EOF
```

### Adım 4.3: Toplu Rapor Üret

```bash
python generate_report.py \
  --results_dir results/outputs/ \
  --output_report analysis_report.html
```

---

## 🎯 5. Hybrid Mode (Rule-based + DL)

Eğer U-Net modelini eğittiyse, VFA/PMA'yı DL ile geliştir:

```bash
python launcher.py \
  --input_data_path "azureml://datastores/workspaceblobstore/paths/amos22-data/" \
  --run_vfa_pma \
  --wait \
  --download \
  --download_dir "./results_hybrid" \
  --use_dl \
  --model_path "./models/outputs/best_unet.pt"
```

---

## 🔍 6. Radyolog Validasyonu

Sonuçları radyolog ölçümleri ile karşılaştır:

```bash
# Ground truth CSV dosyası oluştur
# (Format: case_id, vfa_mm2, pma_mm2, sex, height_m)

python validate_results.py \
  --results_dir results/outputs/ \
  --ground_truth ground_truth.csv \
  --output_metrics validation_metrics.json
```

**Metrikleri kontrol et:**
```bash
python << 'EOF'
import json

with open("validation_metrics.json") as f:
    metrics = json.load(f)

print(f"MAE VFA: {metrics['mae_vfa']:.1f} mm²")
print(f"MAE PMA: {metrics['mae_pma']:.1f} mm²")
print(f"Pearson r (VFA): {metrics['pearson_r_vfa']:.3f}")
print(f"Bias (VFA): {metrics['bias_vfa']:.2f}%")
EOF
```

---

## 📈 7. Toplu İşleme Sonrası

### Adım 7.1: Tüm AMOS22'yi İşle

Başlangıçta test için 50 vaka işlediyse, tümünü işle:

```bash
# Tüm AMOS22 verilerini yükle
az storage blob upload-batch \
  --account-name "$STORAGE_ACCOUNT" \
  --destination amos22-full \
  --source "/path/to/full/AMOS22/"

# Pipeline'ı tam dataset ile çalıştır
python launcher.py \
  --input_data_path "azureml://datastores/workspaceblobstore/paths/amos22-full/" \
  --run_vfa_pma \
  --run_training \
  --wait
```

---

## 🛠️ Sorun Giderme

### Problem: "Workspace not found"

```bash
# Workspace adını kontrol et
az ml workspace list --resource-group "$AZURE_RESOURCE_GROUP"

# Ya da .config dosyasını oluştur
az ml folder attach --resource-group "$AZURE_RESOURCE_GROUP" --workspace-name "$AZURE_ML_WORKSPACE"
```

### Problem: "Quota exceeded"

```bash
# GPU availability'i kontrol et
az ml compute list-skus --location eastus | grep NC4as

# Farklı region dene
export AZURE_REGION="westus2"
./setup_azure_ml.sh
```

### Problem: "Storage account name already taken"

```bash
# Benzersiz bir isim kullan
export AZURE_STORAGE_ACCOUNT="l3sofastg$(date +%s)"
./setup_azure_ml.sh
```

---

## 💰 Maliyet Tahmini

| Kaynak | Birim Fiyatı | Miktar | Toplam |
|--------|--------------|--------|--------|
| T4 GPU (NC4as_T4_v3) | ~$0.35/saat | 5 saat | ~$1.75 |
| Storage | ~$0.024/GB | 100 GB | ~$2.40 |
| **Toplam** | - | - | **~$4/çalıştırma** |

---

## 📚 Diğer Kaynaklar

- [Azure ML Workspace Kurulumu](https://learn.microsoft.com/en-us/azure/machine-learning/concept-workspace)
- [Azure ML Compute Targets](https://learn.microsoft.com/en-us/azure/machine-learning/concept-compute-target)
- [Batch Processing](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-batch-endpoint)

---

## ✅ Checklist

- [ ] Azure subscription aktif
- [ ] Azure CLI login yapıldı (`az account show`)
- [ ] `setup_azure_ml.sh` çalıştırıldı
- [ ] AMOS22 Storage'a yüklendi
- [ ] VFA/PMA pipeline başlatıldı
- [ ] Sonuçlar indirildi
- [ ] Radyolog validasyonu yapıldı
- [ ] U-Net modeli eğitildi (opsiyonel)
- [ ] Hybrid mode test edildi (opsiyonel)

---

**Hazırlandı:** Azure ML L3 SO Analysis Team  
**Son Güncelleme:** Aralık 2025
