# GitHub Actions ile Azure ML Entegrasyonu

Bu klasör, GitHub üzerinden Azure ML'de GPU eğitimi başlatmak için gerekli konfigürasyonları içerir.

## Dosyalar

- **`training-job.yml`**: Azure ML Job tanımı (compute, environment, inputs/outputs)
- **`conda-env.yml`**: Python bağımlılıkları (PyTorch, MONAI, TotalSegmentator, vb.)
- **`run_training.sh`**: Tam pipeline scripti (DICOM → TS/C2C → L3 → 60 epok → QA)

## Kurulum Adımları

### 1. Azure Credentials (Service Principal)

Azure Portal veya CLI ile Service Principal oluştur:

```bash
az ad sp create-for-rbac \
  --name "github-actions-l3" \
  --role contributor \
  --scopes /subscriptions/<subscription-id>/resourceGroups/<resource-group> \
  --sdk-auth
```

Çıktıyı kopyala ve GitHub Secrets'e ekle.

### 2. GitHub Secrets Ekle

GitHub repo → Settings → Secrets and variables → Actions → New repository secret

Eklenecek secret'lar:

| Secret Adı | Değer |
|------------|-------|
| `AZURE_CREDENTIALS` | Service principal JSON çıktısı |
| `AZURE_ML_WORKSPACE` | Azure ML workspace adı (örn: `L3-ML-Workspace`) |
| `AZURE_RESOURCE_GROUP` | Resource group adı |

### 3. Azure ML Workspace ve Compute Hazırla

**Workspace oluştur:**
```bash
az ml workspace create \
  --name L3-ML-Workspace \
  --resource-group <your-rg> \
  --location eastus
```

**GPU Compute Instance oluştur:**
```bash
az ml compute create \
  --name l3-gpu-compute \
  --type AmlCompute \
  --size Standard_NC6s_v3 \
  --min-instances 0 \
  --max-instances 1 \
  --workspace-name L3-ML-Workspace \
  --resource-group <your-rg>
```

### 4. AMOS22 Verilerini Azure Blob'a Yükle

```bash
# Storage account oluştur (yoksa)
az storage account create \
  --name <storage-name> \
  --resource-group <your-rg> \
  --location eastus

# Blob container oluştur
az storage container create \
  --name amos22 \
  --account-name <storage-name>

# Verileri yükle
az storage blob upload-batch \
  --account-name <storage-name> \
  --destination amos22 \
  --source ~/Desktop/amos22
```

**Azure ML Datastore'a kaydet:**
```bash
az ml datastore create \
  --name amos22-datastore \
  --type azure_blob \
  --account-name <storage-name> \
  --container-name amos22 \
  --workspace-name L3-ML-Workspace \
  --resource-group <your-rg>
```

### 5. GitHub'dan Eğitimi Başlat

1. GitHub repo → Actions sekmesi
2. "Azure ML Training Pipeline" workflow'unu seç
3. "Run workflow" → parametreleri ayarla:
   - **compute_target**: `l3-gpu-compute`
   - **epochs**: `60`
   - **preset**: `eval`
   - **use_ts**: `true`
   - **use_c2c**: `false`
4. "Run workflow" tıkla

Workflow başlar, Azure ML'de job oluşturur ve stream eder. Tamamlandığında artifacts GitHub'a yüklenir.

## Workflow Parametreleri

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `compute_target` | Azure ML compute adı | `l3-gpu-compute` |
| `epochs` | Eğitim epoch sayısı | `60` |
| `preset` | İşleme preset'i | `eval` |
| `use_ts` | TotalSegmentator kullan | `true` |
| `use_c2c` | C2C öğretmen kullan | `false` |

## Çıktılar

Eğitim tamamlandığında:

- **Azure ML Studio'da**: Çalışma logları, metrikler, modelller
- **GitHub Actions Artifacts**: QA overlayleri, checkpointler, sonuç JSON'ları

Artifacts'i indirmek için:
```
Actions → Workflow run → Artifacts bölümü → "training-logs-XXX" indir
```

## Manuel Test (Lokal)

Azure CLI ile test etmek için:

```bash
az ml job create \
  --file .github/azure-ml/training-job.yml \
  --workspace-name L3-ML-Workspace \
  --resource-group <your-rg> \
  --stream
```

## Sorun Giderme

**Service principal yetki hatası:**
```bash
az role assignment create \
  --assignee <service-principal-app-id> \
  --role "AzureML Data Scientist" \
  --scope /subscriptions/<subscription-id>/resourceGroups/<rg>/providers/Microsoft.MachineLearningServices/workspaces/L3-ML-Workspace
```

**Datastore bulunamadı:**
- `training-job.yml` içinde `inputs.amos22_data.path` değerini kendi datastore yoluna göre düzenle.

**Compute başlatma hatası:**
- Azure Portal'dan compute durumunu kontrol et; idle ise manuel başlat.

---

## Sonuç

Artık GitHub repo sayfasından tek tıkla Azure ML GPU'da 60 epok eğitim başlatabilirsin. Sonuçlar otomatik olarak Azure'da ve GitHub artifacts'te saklanır.
