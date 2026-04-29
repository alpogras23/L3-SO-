#!/bin/bash
# Azure ML Quick Configuration
# Etkileşimli setup - tüm parametreleri sor

set -e

echo "=========================================="
echo "🔧 Azure ML L3 VFA/PMA Setup Sihirbazı"
echo "=========================================="
echo ""

# Subscription ID
echo "📋 Azure Subscription ID'nizi girin:"
read -p "Subscription ID: " SUBSCRIPTION_ID

# Resource Group
echo ""
echo "📋 Azure Resource Group adını girin:"
read -p "Resource Group (default: l3-so-rg): " RESOURCE_GROUP
RESOURCE_GROUP="${RESOURCE_GROUP:-l3-so-rg}"

# Location
echo ""
echo "📋 Azure bölgesini seçin (default: eastus):"
read -p "Location [eastus/westus2/northeurope]: " LOCATION
LOCATION="${LOCATION:-eastus}"

# Workspace Name
echo ""
echo "📋 Azure ML Workspace adını girin:"
read -p "Workspace Name (default: l3-vfa-pma-workspace): " WORKSPACE_NAME
WORKSPACE_NAME="${WORKSPACE_NAME:-l3-vfa-pma-workspace}"

# Storage Account
echo ""
echo "📋 Storage Account adını girin (3-24 karakter, lowercase):"
read -p "Storage Account (default: l3vfapmastorage): " STORAGE_ACCOUNT
STORAGE_ACCOUNT="${STORAGE_ACCOUNT:-l3vfapmastorage}"

# AMOS22 Path
echo ""
echo "📂 AMOS22 NIfTI dosyalarının yolunu girin:"
read -p "AMOS22 Path (default: /Users/alperenogras/Desktop/L3_SO_ANALYSIS/data/AMOS22/): " AMOS_PATH
AMOS_PATH="${AMOS_PATH:-/Users/alperenogras/Desktop/L3_SO_ANALYSIS/data/AMOS22/}"

# TS Path
echo ""
echo "📂 TotalSegmentator maskelerinin yolunu girin:"
read -p "TS Path (default: /Users/alperenogras/Desktop/L3_SO_ANALYSIS/data/ts_masks/): " TS_PATH
TS_PATH="${TS_PATH:-/Users/alperenogras/Desktop/L3_SO_ANALYSIS/data/ts_masks/}"

# Summary
echo ""
echo "=========================================="
echo "✅ Ayarlar Özeti"
echo "=========================================="
echo "Subscription ID: $SUBSCRIPTION_ID"
echo "Resource Group: $RESOURCE_GROUP"
echo "Location: $LOCATION"
echo "Workspace: $WORKSPACE_NAME"
echo "Storage Account: $STORAGE_ACCOUNT"
echo "AMOS22 Path: $AMOS_PATH"
echo "TS Path: $TS_PATH"
echo ""

# Confirm
read -p "Onaylıyor musunuz? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "❌ İptal edildi"
    exit 0
fi

# Run setup
echo ""
echo "🚀 Setup başlatılıyor..."
cd "$(dirname "$0")"

bash setup_azure_ml.sh \
  "$SUBSCRIPTION_ID" \
  "$RESOURCE_GROUP" \
  "$LOCATION" \
  "$WORKSPACE_NAME" \
  "$STORAGE_ACCOUNT" \
  "l3-gpu-cluster" \
  "Standard_NC4as_T4_v3" \
  "$AMOS_PATH" \
  "$TS_PATH"

echo ""
echo "=========================================="
echo "✅ Setup Tamamlandı!"
echo "=========================================="
echo ""
echo "Sonraki adım - Pipeline'ı başlatmak için:"
echo ""
echo "python launcher.py \\"
echo "  --subscription_id $SUBSCRIPTION_ID \\"
echo "  --resource_group $RESOURCE_GROUP \\"
echo "  --workspace_name $WORKSPACE_NAME \\"
echo "  --scripts_dir ./"
echo ""
