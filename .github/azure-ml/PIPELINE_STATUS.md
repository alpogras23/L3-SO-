# 🚀 PIPELINE BAŞLATILDI - 240 AMOS22 İŞLEME

## ✅ STATUS: RUNNING

**Başlangıç Zamanı**: 2025-12-06 22:36 UTC+3  
**Orchestrator PID**: 99869  
**Log Dosyası**: `/tmp/pipeline_log.txt`

---

## 📌 ADIM 1: TEACHER GENERATION (TotalSegmentator)

| Özellik | Değer |
|---------|-------|
| **Status** | ✅ RUNNING |
| **Job ID** | `shy_roti_nkl78tcjgr` |
| **İşlenen Dosya Sayısı** | **240 (tüm AMOS22)** |
| **Toplam Boyut** | 8.2 GB |
| **Giriş Path** | `/Users/alperenogras/Desktop/amos22/imagesTr/` |
| **Çıktı** | `teacher_labels/` (HU slices + masks) |
| **ETA** | 2-3 saat |
| **Compute** | Azure ML CPU Cluster (2 vCPU) |

### Çıktı Yapısı (Her Case için)
```
teacher_labels/
├── amos_0001/
│   ├── hu_slice.png        (HU intensity visualization)
│   ├── psoas_left.png      (Binary mask)
│   ├── psoas_right.png     (Binary mask)
│   ├── vat.png             (Visceral fat mask)
│   └── metadata.json       (L3 slice index, HU stats)
├── amos_0004/
│   └── ... (same structure)
...
└── amos_0600/
    └── ... (240 cases total)
```

---

## 🔄 ADIM 2: U-NET TRAINING (60 Epoch)

| Özellik | Değer |
|---------|-------|
| **Status** | ⏳ WAITING (Step 1 tamamlandığında otomatik başlar) |
| **Trigger** | Step 1 success |
| **Input** | Teacher labels from Step 1 |
| **Model Architecture** | MONAI 2D U-Net |
| **Channels** | 1 → 32 → 64 → 128 → 256 → 3 |
| **Input Channels** | 1 (HU intensity) |
| **Output Channels** | 3 (psoas_left, psoas_right, vat) |
| **Batch Size** | 4 |
| **Learning Rate** | 0.001 |
| **Epochs** | 60 |
| **Optimizer** | AdamW |
| **Loss Function** | DiceCELoss |
| **Input Size** | 256×256 px |
| **ETA** | 2-4 saat (Step 1 bitince) |
| **Compute** | Azure ML CPU Cluster |

---

## 📊 ADIM 3: VFA/PMA INFERENCE

| Özellik | Değer |
|---------|-------|
| **Status** | ⏳ WAITING (Step 2 tamamlandığında otomatik başlar) |
| **Trigger** | Step 2 success |
| **Input** | Trained checkpoint from Step 2 |
| **Output Metrics** | VFA (mm²), PMA (mm²), Confidence Scores |
| **Output Files** | metrics.json, vfa_pma_report.csv, overlay images |
| **Processing** | 240 cases × L3 slice |
| **ETA** | 10-15 dakika (Step 2 bitince) |

### Üretilecek Sonuçlar
- **VFA (Visceral Fat Area)**: mm² cinsinden ölçüm
- **PMA (Psoas Muscle Area)**: mm² cinsinden ölçüm
- **Confidence Scores**: Model güven skorları
- **Overlay Images**: Segmentasyon ile görseller
- **CSV Report**: Tüm 240 case için tablosal sonuçlar

---

## ⏱️ TIMELINE

```
ŞİMDİ (22:36)
    ↓ (2-3 saat)
Adım 2 başlıyor (~00:36-01:36 sonraki gün)
    ↓ (2-4 saat)
Adım 3 başlıyor (~02:36-05:36 sonraki gün)
    ↓ (10-15 dakika)
🎉 PIPELINE COMPLETE (~02:50-05:50 sonraki gün)
```

**Beklenen Toplam Süre**: 7-8 saat

---

## 📋 MONITORING

### Real-time Log (Continuous)
```bash
tail -f /tmp/pipeline_log.txt
```

### Manual Status Check
```bash
# Step 1 Status
az ml job show --name shy_roti_nkl78tcjgr --query status

# View all jobs
https://ml.azure.com/runs
```

### Job Details
- **Step 1 Job ID**: `shy_roti_nkl78tcjgr`
- **Step 2 Job ID**: Auto-generated (when Step 1 completes)
- **Step 3 Job ID**: Auto-generated (when Step 2 completes)

---

## 🎯 HEDEFİ

**Amaç**: "amos 22 ts üzerinden bunları segmente et. 60 epoch eğitimleri tamamla. en iyi en doğru vfa pma hesabına ulaş"

**Yapılıyor**:
1. ✅ **TotalSegmentator** → 240 AMOS22 CT segmentasyonu
2. ✅ **MONAI U-Net** → 60 epoch training (teacher labels üzerinde)
3. ✅ **Inference** → VFA/PMA hesaplama (tüm 240 case)

**Veri**:
- **Giriş**: 240 AMOS22 CT dosyası (8.2 GB)
- **Manifest**: `data/amos_manifest_full.csv` (240 entries)
- **Çıktı**: VFA/PMA metrics + overlay images

---

## ✅ OTOMASYÖN DETAYLARI

**Auto-Orchestration Script**: `/Users/alperenogras/Desktop/L3_SO_ANALYSIS/.github/azure-ml/auto_pipeline_v3.sh`

### Otomatik Zincirlemeler:
- Step 1 tamamlandığında → Step 2 otomatik başlar
- Step 2 tamamlandığında → Step 3 otomatik başlar
- Tüm adımlar başarı kontrol ile güvenli

### Manuel Müdahale Gerekmez
- ✅ Orchestrator arka planda çalışıyor (PID: 99869)
- ✅ Tüm adımlar otomatik bağlanıyor
- ✅ Hata kontrolü var, başarısızlık durumunda bildirim

---

## 📁 ÇIKTI KONUMLARI

```
azureml://jobs/shy_roti_nkl78tcjgr/outputs/teacher_labels/
    ├─ [case_id]/hu_slice.png
    ├─ [case_id]/psoas_left.png
    ├─ [case_id]/psoas_right.png
    ├─ [case_id]/vat.png
    └─ [case_id]/metadata.json
    × 240 cases

azureml://jobs/[step2_job_id]/outputs/model/
    └─ checkpoint.pth (trained U-Net)

azureml://jobs/[step3_job_id]/outputs/results/
    ├─ metrics.json
    ├─ vfa_pma_report.csv
    └─ overlays/ (visualization)
```

---

**Son Güncelleme**: 2025-12-06 22:36 UTC+3
