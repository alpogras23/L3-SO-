#!/usr/bin/env python3
"""
Azure ML Workspace & Compute oluştur - Python SDK kullanarak
"""

import json
import time
from pathlib import Path

def setup_azureml_workspace():
    """Azure ML SDK ile workspace ve compute oluştur"""
    
    print("\n" + "="*70)
    print("🚀 AZURE ML WORKSPACE & COMPUTE SETUP (Python SDK)")
    print("="*70 + "\n")
    
    try:
        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import ComputeInstance
        from azure.identity import DefaultAzureCredential
    except ImportError as e:
        print(f"❌ Import hatası: {e}")
        print("📦 Yükleniyor: pip install azure-ai-ml azure-identity")
        import subprocess
        subprocess.run([
            "pip", "install", "-q", 
            "azure-ai-ml", "azure-identity"
        ], check=True)
        # Tekrar import et
        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import ComputeInstance
        from azure.identity import DefaultAzureCredential
    
    # Konfigürasyon
    SUBSCRIPTION_ID = "d2d593d8-01af-4128-98e5-31ede386359f"
    RESOURCE_GROUP = "l3-rg"
    WORKSPACE_NAME = "l3-vfa-pma-ws"
    WORKSPACE_LOCATION = "westeurope"
    COMPUTE_NAME = "l3-compute-01"
    
    print(f"📝 Yapılandırma:")
    print(f"   • Workspace: {WORKSPACE_NAME}")
    print(f"   • Resource Group: {RESOURCE_GROUP}")
    print(f"   • Location: {WORKSPACE_LOCATION}")
    print(f"   • Compute: {COMPUTE_NAME}")
    print()
    
    try:
        # Credentials al
        print("1️⃣  Azure credentials alınıyor...")
        credential = DefaultAzureCredential()
        
        # ML Client oluştur
        print("2️⃣  ML Client bağlanıyor...")
        ml_client = MLClient(
            credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME
        )
        
        # Workspace bilgisini al
        print("3️⃣  Workspace bilgisi alınıyor...")
        try:
            ws = ml_client.workspaces.get(name=WORKSPACE_NAME)
            print(f"   ✅ Workspace bulundu: {ws.name}")
        except Exception as e:
            print(f"   ⚠️  Workspace oluşturulmamış: {e}")
            print("   💡 Azure Portal'dan veya önceki script'ten oluşturun")
            return False
        
        # Compute Instance oluştur
        print("4️⃣  Compute Instance oluşturuluyor...")
        try:
            compute = ml_client.compute.get(name=COMPUTE_NAME)
            print(f"   ✅ Compute bulundu: {compute.name}")
        except:
            print(f"   ℹ️  Compute oluşturuluyor...")
            compute_config = ComputeInstance(
                name=COMPUTE_NAME,
                size="Standard_D2s_v3",
                idle_time_before_shutdown_minutes=30
            )
            ml_client.compute.begin_create_or_update(compute_config).result()
            print(f"   ✅ Compute oluşturuldu: {COMPUTE_NAME}")
        
        # Config dosyasını güncelle
        print("5️⃣  Config dosyası güncelleniyor...")
        config_path = Path(__file__).parent / "azure_ml_config.json"
        
        config = {
            "subscription_id": SUBSCRIPTION_ID,
            "resource_group": RESOURCE_GROUP,
            "workspace_name": WORKSPACE_NAME,
            "workspace_location": WORKSPACE_LOCATION,
            "compute_name": COMPUTE_NAME,
            "storage_account": "l3storage001",
            "storage_container": "amos22-data",
            "ml_client_ready": True
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   ✅ Config kaydedildi: {config_path}")
        
        # Özet
        print("\n" + "="*70)
        print("✅ AZURE ML SETUP BAŞARILI!")
        print("="*70)
        print(f"""
📊 Kurulu Kaynaklar:
   • Workspace: {WORKSPACE_NAME}
   • Resource Group: {RESOURCE_GROUP}
   • Location: {WORKSPACE_LOCATION}
   • Compute Instance: {COMPUTE_NAME}

📁 Config dosyası: azure_ml_config.json

🚀 Sonraki Adım:
   1. AMOS22 datasını Azure Storage'a yükle
   2. launcher.py --run_vfa_pma --wait
""")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = setup_azureml_workspace()
    sys.exit(0 if success else 1)
