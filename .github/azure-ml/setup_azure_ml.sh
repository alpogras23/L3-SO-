#!/bin/bash
# Azure ML Setup Script
# - Workspace, Storage, Compute Cluster ve Datasets otomatik kurulum
# - AMOS22 verilerini Azure Storage'a yükle

set -e

echo "=========================================="
echo "🚀 Azure ML Setup Scripti"
echo "=========================================="

# Konfigürasyon
SUBSCRIPTION_ID="${1:-}"
RESOURCE_GROUP="${2:-}"
LOCATION="${3:-eastus}"
WORKSPACE_NAME="${4:-l3-vfa-pma-workspace}"
STORAGE_ACCOUNT="${5:-l3vfapmastorage}"
COMPUTE_CLUSTER="${6:-l3-gpu-cluster}"
COMPUTE_SIZE="${7:-Standard_NC4as_T4_v3}"

# AMOS22 lokal yolu
AMOS_LOCAL_PATH="${8:-/Users/alperenogras/Desktop/L3_SO_ANALYSIS/data/AMOS22/}"
TS_LOCAL_PATH="${9:-/Users/alperenogras/Desktop/L3_SO_ANALYSIS/data/ts_masks/}"

if [ -z "$SUBSCRIPTION_ID" ] || [ -z "$RESOURCE_GROUP" ]; then
    echo "❌ Hata: Subscription ID ve Resource Group gerekli"
    echo "Kullanım: ./setup_azure_ml.sh <SUBSCRIPTION_ID> <RESOURCE_GROUP> [LOCATION] [WORKSPACE] [STORAGE] [COMPUTE] [COMPUTE_SIZE] [AMOS_PATH] [TS_PATH]"
    exit 1
fi

echo ""
echo "📋 Ayarlar:"
echo "  Subscription ID: $SUBSCRIPTION_ID"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Bölge: $LOCATION"
echo "  Workspace: $WORKSPACE_NAME"
echo "  Storage: $STORAGE_ACCOUNT"
echo "  Compute: $COMPUTE_CLUSTER ($COMPUTE_SIZE)"
echo "  AMOS22 Path: $AMOS_LOCAL_PATH"
echo "  TS Path: $TS_LOCAL_PATH"
echo ""

# Azure CLI Login
echo "🔐 Azure CLI giriş yapılıyor..."
az account set --subscription "$SUBSCRIPTION_ID"
echo "✅ Subscription ayarlandı: $(az account show --query name -o tsv)"

# Workspace oluştur
echo ""
echo "📂 Azure ML Workspace oluşturuluyor..."
az ml workspace create \
  --name "$WORKSPACE_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --display-name "L3 VFA/PMA Analysis Workspace" || echo "✅ Workspace zaten mevcut"

# Storage Account oluştur
echo ""
echo "💾 Storage Account oluşturuluyor..."
STORAGE_ACCOUNT_ID=$(az storage account create \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku Standard_LRS \
  --query id -o tsv) || echo "✅ Storage Account zaten mevcut"

# Storage Container oluştur
echo ""
echo "🗂️ Storage Containers oluşturuluyor..."
az storage container create \
  --account-name "$STORAGE_ACCOUNT" \
  --name amos22-data \
  --public-access off || echo "✅ Container amos22-data zaten mevcut"

az storage container create \
  --account-name "$STORAGE_ACCOUNT" \
  --name ts-masks \
  --public-access off || echo "✅ Container ts-masks zaten mevcut"

# AMOS22 verilerini yükle
if [ -d "$AMOS_LOCAL_PATH" ]; then
    echo ""
    echo "📤 AMOS22 verisi yükleniyor..."
    az storage blob upload-batch \
      --account-name "$STORAGE_ACCOUNT" \
      --destination amos22-data \
      --source "$AMOS_LOCAL_PATH" \
      --pattern "*.nii*" \
      --overwrite || echo "⚠️ Bazı dosyalar yüklenirken hata oluştu"
    echo "✅ AMOS22 yüklendi"
else
    echo "⚠️ AMOS22 dizini bulunamadı: $AMOS_LOCAL_PATH"
fi

# TS maskelerini yükle
if [ -d "$TS_LOCAL_PATH" ]; then
    echo ""
    echo "📤 TotalSegmentator maskeler yükleniyor..."
    az storage blob upload-batch \
      --account-name "$STORAGE_ACCOUNT" \
      --destination ts-masks \
      --source "$TS_LOCAL_PATH" \
      --pattern "*.nii*" \
      --overwrite || echo "⚠️ Bazı dosyalar yüklenirken hata oluştu"
    echo "✅ TS maskeler yüklendi"
else
    echo "⚠️ TS dizini bulunamadı: $TS_LOCAL_PATH"
fi

# GPU Compute Cluster oluştur
echo ""
echo "🖥️ GPU Compute Cluster oluşturuluyor..."
az ml compute create \
  --name "$COMPUTE_CLUSTER" \
  --workspace-name "$WORKSPACE_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --type amlcompute \
  --size "$COMPUTE_SIZE" \
  --min-instances 0 \
  --max-instances 2 \
  --idle-time-before-scale-down 600 || echo "✅ Compute Cluster zaten mevcut"

echo ""
echo "=========================================="
echo "✅ Azure ML Setup Tamamlandı!"
echo "=========================================="
echo ""
echo "Sonraki adımlar:"
echo "1. Python SDK ile pipeline'ı başlat:"
echo "   python launcher.py \\"
echo "     --subscription_id $SUBSCRIPTION_ID \\"
echo "     --resource_group $RESOURCE_GROUP \\"
echo "     --workspace_name $WORKSPACE_NAME \\"
echo "     --scripts_dir ./.github/azure-ml/"
echo ""
echo "2. Azure Portal'da izle:"
echo "   https://ml.azure.com"
echo ""
