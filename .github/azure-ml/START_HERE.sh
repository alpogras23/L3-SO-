#!/bin/bash
# BAŞLATMA REHBERI - Azure ML L3 VFA/PMA Analysis Pipeline
# 
# Bu script, tüm Azure ML setup ve pipeline'ı başlatmak için
# adım adım rehberdir.

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   🚀 Azure ML L3 VFA/PMA Analysis Pipeline Setup Guide    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Adım 1: Azure CLI Doğrulama
echo "📋 ADIM 1: Azure CLI Kurulumu Kontrol Ediliyor..."
echo ""
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI bulunamadı!"
    echo "📦 Kurulum: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi
echo "✅ Azure CLI kurulu: $(az --version | head -1)"
echo ""

# Adım 2: Python Ortamı
echo "📋 ADIM 2: Python Ortamı Kontrol Ediliyor..."
echo ""
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 bulunamadı!"
    exit 1
fi
echo "✅ Python kurulu: $(python3 --version)"
echo ""

# Adım 3: Required Python Packages
echo "📋 ADIM 3: Gerekli Python Paketleri Yükleniyor..."
echo ""
python3 -m pip install -q azureml-sdk || {
    echo "❌ azureml-sdk yükleme başarısız"
    echo "💡 Çözüm: pip install azureml-sdk"
    exit 1
}
echo "✅ Required packages kurulu"
echo ""

# Adım 4: Azure Credentials
echo "📋 ADIM 4: Azure Giriş Yapılıyor..."
echo ""
az account show > /dev/null 2>&1 || {
    echo "❌ Azure'a giriş yapılmamış!"
    echo "🔐 Giriş yapılıyor..."
    az login
}
SUBSCRIPTION=$(az account show --query id -o tsv)
echo "✅ Giriş başarılı!"
echo "   Subscription: $SUBSCRIPTION"
echo ""

# Adım 5: Quick Setup Seçimi
echo "📋 ADIM 5: Setup Yöntemi Seçimi"
echo ""
echo "1️⃣  Etkileşimli Setup (önerilen - tüm parametreleri sorar)"
echo "2️⃣  Manuel Setup (komut satırında parametreler)"
echo "3️⃣  Konfigürasyon Dosyasından (azure-ml.config.json)"
echo ""
read -p "Seçiminiz (1-3): " CHOICE

case $CHOICE in
    1)
        echo ""
        echo "🔧 Etkileşimli Setup başlatılıyor..."
        bash quick_setup.sh
        ;;
    2)
        echo ""
        echo "📝 Manuel setup için parametreler gir:"
        read -p "Resource Group: " RG
        read -p "Workspace Name: " WS
        read -p "Subscription ID: " SUB
        bash setup_azure_ml.sh "$SUB" "$RG"
        ;;
    3)
        echo ""
        echo "❌ Config dosyasından setup henüz uygulanmadı"
        echo "💡 Lütfen manuel veya etkileşimli setup'ı seçin"
        exit 1
        ;;
    *)
        echo "❌ Geçersiz seçim"
        exit 1
        ;;
esac

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    ✅ Setup Tamamlandı!                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Sonraki Adımlar:"
echo ""
echo "1️⃣  Pipeline'ı Başlat:"
echo "    python launcher.py \\"
echo "      --subscription_id <ID> \\"
echo "      --resource_group <RG> \\"
echo "      --workspace_name <WS> \\"
echo "      --scripts_dir ./"
echo ""
echo "2️⃣  Azure Portal'da İzle:"
echo "    https://ml.azure.com"
echo ""
echo "3️⃣  Run tamamlandığında sonuçları indir:"
echo "    az ml run download --run-id <RUN_ID>"
echo ""
echo "📚 Daha fazla bilgi:"
echo "    cat README_PIPELINE.md"
echo ""
