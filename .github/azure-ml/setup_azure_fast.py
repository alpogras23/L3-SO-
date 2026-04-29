#!/usr/bin/env python3
"""
HIZLI Azure ML Setup - Resource Group, Workspace, Storage, Compute oluştur
"""

import json
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd, description=""):
    """Azure CLI komutunu çalıştır ve çıktı döndür"""
    print(f"▶️  {description}... " if description else f"▶️  {cmd[:60]}...")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ Hata: {e.stderr}")
        return None

def main():
    print("\n" + "="*70)
    print("🚀 AZURE ML L3 VFA/PMA HIZLI SETUP")
    print("="*70 + "\n")
    
    # Konfigürasyon
    SUBSCRIPTION_ID = "d2d593d8-01af-4128-98e5-31ede386359f"
    RESOURCE_GROUP = "l3-rg"
    WORKSPACE_NAME = "l3-vfa-pma-ws"
    LOCATION = "westeurope"
    COMPUTE_NAME = "l3-compute-01"
    STORAGE_ACCOUNT = "l3storage001"
    CONTAINER_NAME = "amos22-data"
    
    print(f"📝 Yapılandırma:")
    print(f"   • Subscription: {SUBSCRIPTION_ID[:20]}...")
    print(f"   • Resource Group: {RESOURCE_GROUP}")
    print(f"   • Workspace: {WORKSPACE_NAME}")
    print(f"   • Location: {LOCATION}")
    print(f"   • Compute: {COMPUTE_NAME}")
    print(f"   • Storage: {STORAGE_ACCOUNT}")
    print(f"   • Container: {CONTAINER_NAME}")
    print()
    
    # Subscription ayarla
    print("1️⃣  Subscription ayarlanıyor...")
    run_cmd(
        f"az account set --subscription {SUBSCRIPTION_ID}",
        "Subscription ayarlanıyor"
    )
    
    # Resource Group oluştur
    print("2️⃣  Resource Group oluşturuluyor...")
    run_cmd(
        f"az group create --name {RESOURCE_GROUP} --location {LOCATION}",
        f"Resource Group '{RESOURCE_GROUP}' oluşturuluyor"
    )
    
    # Storage Account oluştur
    print("3️⃣  Storage Account oluşturuluyor...")
    run_cmd(
        f"az storage account create "
        f"--resource-group {RESOURCE_GROUP} "
        f"--name {STORAGE_ACCOUNT} "
        f"--location {LOCATION} "
        f"--sku Standard_LRS",
        f"Storage Account '{STORAGE_ACCOUNT}' oluşturuluyor"
    )
    
    # Storage Connection String al
    print("4️⃣  Storage connection string alınıyor...")
    conn_str = run_cmd(
        f"az storage account show-connection-string "
        f"--resource-group {RESOURCE_GROUP} "
        f"--name {STORAGE_ACCOUNT} "
        f"--query connectionString -o tsv"
    )
    if not conn_str:
        print("❌ Connection string alınamadı!")
        return False
    
    # Container oluştur
    print("5️⃣  Blob container oluşturuluyor...")
    run_cmd(
        f"az storage container create "
        f"--account-name {STORAGE_ACCOUNT} "
        f"--name {CONTAINER_NAME}",
        f"Container '{CONTAINER_NAME}' oluşturuluyor"
    )
    
    # Azure ML Workspace oluştur
    print("6️⃣  Azure ML Workspace oluşturuluyor...")
    run_cmd(
        f"az ml workspace create "
        f"--resource-group {RESOURCE_GROUP} "
        f"--name {WORKSPACE_NAME} "
        f"--location {LOCATION}",
        f"Workspace '{WORKSPACE_NAME}' oluşturuluyor"
    )
    
    # Workspace config indis
    print("7️⃣  Workspace config indiriliyoor...")
    run_cmd(
        f"az ml workspace download-config "
        f"--resource-group {RESOURCE_GROUP} "
        f"--name {WORKSPACE_NAME} "
        f"--output-file ./.ml/config.json",
        "Config indiriliyyor"
    )
    
    # Compute Instance oluştur
    print("8️⃣  Compute Instance oluşturuluyor...")
    run_cmd(
        f"az ml compute create "
        f"--resource-group {RESOURCE_GROUP} "
        f"--workspace-name {WORKSPACE_NAME} "
        f"--name {COMPUTE_NAME} "
        f"--type ComputeInstance "
        f"--size Standard_D2s_v3",
        f"Compute Instance '{COMPUTE_NAME}' oluşturuluyor"
    )
    
    # Config dosyasını kaydet
    print("9️⃣  Azure ML config dosyası yazılıyor...")
    config_path = Path(__file__).parent / "azure_ml_config.json"
    config = {
        "subscription_id": SUBSCRIPTION_ID,
        "resource_group": RESOURCE_GROUP,
        "workspace_name": WORKSPACE_NAME,
        "workspace_location": LOCATION,
        "storage_account": STORAGE_ACCOUNT,
        "storage_container": CONTAINER_NAME,
        "storage_connection_string": conn_str,
        "compute_name": COMPUTE_NAME,
        "created_at": subprocess.run("date -u +%Y-%m-%dT%H:%M:%SZ", shell=True, capture_output=True, text=True).stdout.strip()
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Config kaydedildi: {config_path}")
    
    # Özet
    print("\n" + "="*70)
    print("✅ SETUP BAŞARILI!")
    print("="*70)
    print(f"""
📊 Oluşturulan Kaynaklar:
   • Resource Group: {RESOURCE_GROUP}
   • Workspace: {WORKSPACE_NAME}
   • Storage Account: {STORAGE_ACCOUNT}
   • Container: {CONTAINER_NAME}
   • Compute Instance: {COMPUTE_NAME}

📁 Config dosyası: azure_ml_config.json

🚀 Sonraki Adım:
   python launcher.py --run_vfa_pma --wait
""")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
