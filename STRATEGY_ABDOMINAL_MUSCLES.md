# 🎯 YENİ HATASIZ L3 PIPELINE STRATEJİSİ

## 📋 SORUN ANALİZİ

### ❌ Önceki Hatalar
1. **TotalSegmentator task="total"** → Psoas mask üretmiyor
2. **fast=True/False** → Task seçimi önemliydi, fast değil
3. **240 case + 60 epoch** → Test olmadan direkt production (riskli)

### ✅ Çözüm
**TotalSegmentator `task="abdominal_muscles"`** kullan:
- Label 19: `psoas_major_right`
- Label 20: `psoas_major_left`
- Toplam 22 abdominal kas (rectus, oblique, erector spinae vb.)

---

## 🚀 YENİ PIPELINE ADIMLARI

### **ADIM 1: TEST - Psoas Doğrulama (ŞUAN ÇALIŞIYOR)**

**Job**: `placid_circle_qwk3zp5qv7`
**Portal**: https://ml.azure.com/runs/placid_circle_qwk3zp5qv7

**Konfigürasyon**:
- Cases: 10
- Epochs: 10
- Batch size: 2
- Task: `abdominal_muscles` ✅

**Beklenen Süre**: ~45-60 dakika
- Step 1: ~20 dakika (10 case)
- Step 2: ~30 dakika (10 epoch)

**Doğrulama**:
```bash
# Job tamamlandığında kontrol:
az ml job show --name placid_circle_qwk3zp5qv7 --resource-group l3-rg --workspace-name l3-vfa-pma-ws

# Logları indir:
az ml job download --name placid_circle_qwk3zp5qv7 --download-path ./test_output

# Psoas mask kontrolü:
grep "psoas_major" ./test_output/artifacts/user_logs/std_log.txt
```

**Başarı Kriterleri**:
- ✅ `psoas_major_left` BULUNDU
- ✅ `psoas_major_right` BULUNDU
- ✅ Training başladı
- ✅ Loss azalıyor

---

### **ADIM 2: PRODUCTION - Full Training (Test Başarılı ise)**

**EĞER** test başarılı ise → Full training başlat:

```bash
cd /Users/alperenogras/Desktop/L3_SO_ANALYSIS
source .venv/bin/activate

az ml job create \
  --file .github/azure-ml/job_combined_step1_step2.yml \
  --resource-group l3-rg \
  --workspace-name l3-vfa-pma-ws
```

**Konfigürasyon**:
- Cases: 240 (full dataset)
- Epochs: 60
- Batch size: 4
- Task: `abdominal_muscles` ✅
- Compute: l3-train-cpu (16GB RAM)

**Beklenen Süre**: ~8-12 saat
- Step 1: ~4-6 saat (240 case)
- Step 2: ~4-6 saat (60 epoch)

---

### **ADIM 3: INFERENCE - Model Test**

Production training tamamlandığında:

```bash
# Model ID'yi al
PROD_JOB_ID="<production_job_id>"

# Inference job başlat
az ml job create \
  --file .github/azure-ml/job_step3_inference.yml \
  --resource-group l3-rg \
  --workspace-name l3-vfa-pma-ws \
  --set inputs.trained_model=azureml://jobs/$PROD_JOB_ID/outputs/trained_model
```

**Çıktı**:
- VFA/PMA ölçümleri (JSON)
- Overlay PNG görüntüleri
- Confidence skorları

---

## 📊 MEVCUT DURUM

### ✅ Tamamlanan
1. ~~TotalSegmentator task problemi tespit edildi~~
2. ~~`abdominal_muscles` task ile fix yapıldı~~
3. ~~Test job başlatıldı (placid_circle_qwk3zp5qv7)~~

### ⏳ Beklemede
1. **Test job tamamlanması** (~45-60 dakika)
2. Psoas mask doğrulaması
3. Production training kararı

### 📝 Sonraki Adımlar
1. **ŞİMDİ**: Test job'u izle (30-60 dk)
2. **SONRA**: Psoas mask var mı kontrol et
3. **BAŞARILI İSE**: Production job başlat (240 case)
4. **BAŞARISIZ İSE**: Alternative strategy (Manuel HU-based psoas detection)

---

## 🔍 İZLEME KOMUTLARI

### Test Job Durumu
```bash
source /Users/alperenogras/Desktop/L3_SO_ANALYSIS/.venv/bin/activate

# Status kontrol
az ml job show --name placid_circle_qwk3zp5qv7 \
  --resource-group l3-rg \
  --workspace-name l3-vfa-pma-ws \
  --query "{status: status}" -o json

# Log stream (canlı takip)
az ml job stream --name placid_circle_qwk3zp5qv7 \
  --resource-group l3-rg \
  --workspace-name l3-vfa-pma-ws
```

### Psoas Doğrulama (Job tamamlandığında)
```bash
# Outputs indir
az ml job download --name placid_circle_qwk3zp5qv7 \
  --download-path ./test_abdominal_output \
  --all

# Psoas mask kontrolü
grep -E "psoas_major|Teacher labels extracted" \
  ./test_abdominal_output/artifacts/user_logs/std_log.txt

# Training metrics
cat ./test_abdominal_output/named-outputs/trained_model/training_history.json
```

---

## 💡 BACKUP PLAN (Eğer Test Başarısız Olursa)

### Plan B: Manuel HU-Based Psoas Detection
Eğer `abdominal_muscles` task de çalışmazsa:

1. Kendi psoas detection algoritması yaz
2. HU thresholding + morphological operations
3. Anatomik konum bilgisi kullan (vertebra çevresinde)
4. Rule-based psoas segmentation

### Plan C: Hybrid Approach
1. TotalSegmentator vertebra detection
2. Manuel psoas region proposal
3. U-Net refinement

---

## 📈 BAŞARI METRİKLERİ

### Step 1 Başarı Kriterleri
- ✅ 10/10 case işlendi
- ✅ Psoas left + right mask üretildi
- ✅ VAT mask doğru (fallback gerekmedi)
- ✅ 0 hata

### Step 2 Başarı Kriterleri
- ✅ Training başladı
- ✅ Loss düzenli azalıyor
- ✅ Val loss < 1.0
- ✅ No NaN/Inf değerler

### Production Başarı Kriterleri
- ✅ 240 case psoas mask'i
- ✅ Val loss < 0.85
- ✅ Model generalize ediyor
- ✅ Inference çalışıyor

---

## 🎯 SONUÇ

**ŞU AN**: Test job çalışıyor (`placid_circle_qwk3zp5qv7`)

**BEKLEME SÜRESİ**: ~45-60 dakika

**SONRASI**: 
- ✅ Başarılı → Production (240 case, 60 epoch)
- ❌ Başarısız → Backup plan (Manuel psoas detection)

---

**Son Güncelleme**: 11 Aralık 2025
**Test Job ID**: placid_circle_qwk3zp5qv7
**Status**: Running
**Portal**: https://ml.azure.com/runs/placid_circle_qwk3zp5qv7
