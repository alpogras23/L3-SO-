#!/bin/bash
# GPU Quota Request Helper Script
# Azure Portal'dan GPU quota artırımı için gerekli bilgiler

cat << 'EOF'

╔══════════════════════════════════════════════════════════════════════╗
║              🎮 GPU QUOTA ARTIRIM REHBERİ 🎮                         ║
╚══════════════════════════════════════════════════════════════════════╝

📊 MEVCUT DURUM
═══════════════════════════════════════════════════════════════════════

✅ NC Family Quota: 6 vCPU (mevcut)
❌ NCSv3, NCASv3_T4, A100, H100: 0 vCPU quota

Region: westeurope
Subscription: d2d593d8-01af-4128-98e5-31ede386359f

🎯 QUOTA ARTIRIM SEÇENEKLERİ
═══════════════════════════════════════════════════════════════════════

Option 1: Azure Portal (Önerilen - 24-48 saat)
──────────────────────────────────────────────────────────
1. https://portal.azure.com açın
2. "Quotas" arayın veya:
   https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade/~/myQuotas

3. "Machine Learning Service" → "Compute" seçin

4. Region: westeurope

5. Aşağıdaki quota'lardan birini seçip "Request Quota Increase":
   
   GPU Options (Fastest to Slowest, Cheapest to Most Expensive):
   
   ┌─────────────────────────────────────────────────────────┐
   │ A) Standard NCASv3_T4 Family (NVIDIA T4 - Önerilen)    │
   │    Size: Standard_NC4as_T4_v3 (4 vCPU, 1x T4 GPU)      │
   │    Request: 8 vCPU quota                                │
   │    Cost: ~$0.52/hour                                    │
   │    Training Time: ~1-2 hours (60 epochs)               │
   └─────────────────────────────────────────────────────────┘
   
   ┌─────────────────────────────────────────────────────────┐
   │ B) Standard NCSv3 Family (NVIDIA V100 - Powerful)      │
   │    Size: Standard_NC6s_v3 (6 vCPU, 1x V100 GPU)        │
   │    Request: 12 vCPU quota                               │
   │    Cost: ~$3.06/hour                                    │
   │    Training Time: ~30-45 min (60 epochs)               │
   └─────────────────────────────────────────────────────────┘
   
   ┌─────────────────────────────────────────────────────────┐
   │ C) Standard NCADSA100v4 Family (NVIDIA A100 - Best)    │
   │    Size: Standard_NC24ads_A100_v4 (24 vCPU, 1x A100)   │
   │    Request: 24 vCPU quota                               │
   │    Cost: ~$3.67/hour                                    │
   │    Training Time: ~15-20 min (60 epochs)               │
   └─────────────────────────────────────────────────────────┘

6. Justification:
   "Machine Learning model training for medical imaging analysis (L3 VFA/PMA 
   segmentation). Training MONAI U-Net for 60 epochs on AMOS22 dataset. 
   Research project requiring GPU acceleration."

7. Yeni Limit: 8-24 vCPU (yukarıdaki seçime göre)

8. Submit Request

Option 2: Farklı Region Deneyin (Hemen)
──────────────────────────────────────────────────────
Bazı regionlarda GPU quota varsayılan olarak açık olabilir:

# Try North Europe (yakın)
az group create --name l3-rg-ne --location northeurope
az ml workspace create --name l3-ws-ne --resource-group l3-rg-ne --location northeurope

# Try East US (genellikle daha fazla quota)
az group create --name l3-rg-eus --location eastus
az ml workspace create --name l3-ws-eus --resource-group l3-rg-eus --location eastus

Option 3: Spot/Low-Priority GPU (Anında - %80 İndirim!)
──────────────────────────────────────────────────────
Low-priority GPU quota genellikle açık (-1 = unlimited):
- Standard NC Family LowPriority: Unlimited
- Cost: ~$0.10/hour (vs $0.52/hour normal)
- Risk: Job interrupted edilebilir (nadir)

Az ml job create ile --set compute.priority=Spot kullanın

Option 4: CPU ile Devam (Mevcut - Yavaş ama İşler)
──────────────────────────────────────────────────────
Mevcut l3-cpu-cluster (STANDARD_D2S_V3) kullanılabilir:
- 2 vCPU, no GPU
- Training time: ~2-4 hours (60 epochs)
- Cost: ~$0.10/hour
- Stable, guaranteed completion

⏱️ SÜRELERİN KARŞILAŞTIRILMASI (60 epoch)
═══════════════════════════════════════════════════════════════════════

CPU (D2s_v3):           ~2-4 hours     [$0.40-0.80]   ✅ Available now
GPU T4:                 ~1-2 hours     [$0.52-1.04]   ⏳ Need quota
GPU V100:               ~30-45 min     [$1.53-2.30]   ⏳ Need quota
GPU A100:               ~15-20 min     [$0.92-1.23]   ⏳ Need quota
Low-Priority T4:        ~1-2 hours     [$0.10-0.20]   ✅ May work now

🚀 HEMEN BAŞLAMAK İÇİN (GPU QUOTA BEKLERKEN)
═══════════════════════════════════════════════════════════════════════

Plan A: Low-Priority GPU ile Dene (Risk: interruption)
────────────────────────────────────────────────────────
cd .github/azure-ml
# job_step2_training.yml'de compute ekleyin:
# compute:
#   type: amlcompute
#   target: azureml:l3-cpu-cluster
#   priority: Spot  # Low-priority!

Plan B: CPU ile Başlat, GPU Gelince Transfer
────────────────────────────────────────────────────────
1. CPU ile training başlat (şimdi)
2. GPU quota gelince:
   - Training checkpoint'inden devam et
   - Kalan epoch'ları GPU'da bitir

Plan C: Colab Pro+ GPU (Alternatif Cloud)
────────────────────────────────────────────────────────
- Anında GPU erişimi (V100/A100)
- $10/month unlimited
- notebooks/colab_pro_plus_L3_training.ipynb hazır
- TotalSegmentator + MONAI + 60 epoch ready

📞 QUOTA REQUEST TRACKING
═══════════════════════════════════════════════════════════════════════

Request Status:
https://portal.azure.com/#view/Microsoft_Azure_Support/HelpAndSupportBlade/~/overview

Typical Response Time: 24-48 hours (business days)
Priority Support: 1-4 hours (enterprise subscriptions)

✅ QUOTA GELDIĞINDE YAPILACAKLAR
═══════════════════════════════════════════════════════════════════════

1. GPU Cluster Oluştur:

az ml compute create \
  --name l3-gpu-t4 \
  --resource-group l3-rg \
  --workspace-name l3-vfa-pma-ws \
  --type AmlCompute \
  --size STANDARD_NC4as_T4_v3 \
  --min-instances 0 \
  --max-instances 1 \
  --idle-time-before-scale-down 1800

2. Job YAML'larını Güncelle:

# job_step2_training.yml
compute: azureml:l3-gpu-t4  # CPU yerine GPU

3. Pipeline'ı Çalıştır:

python run_full_pipeline.py

═══════════════════════════════════════════════════════════════════════

💡 ÖNERİ: Quota request gönderirken CPU ile training başlatın.
   GPU geldiğinde zaten teacher labels hazır olur, sadece training
   tekrar çalıştırılır (çok daha hızlı).

Portal: https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade

═══════════════════════════════════════════════════════════════════════

EOF
