#!/usr/bin/env python3
"""
Azure ML Pipeline Launcher - Azure AI ML SDK kullanarak
VFA/PMA işlemesini başlatır
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

def launch_vfa_pma_job():
    """Azure ML Job olarak VFA/PMA işlemesini başlat"""
    
    print("\n" + "="*70)
    print("🚀 AZURE ML L3 VFA/PMA PIPELINE BAŞLATILIYOR")
    print("="*70 + "\n")
    
    try:
        from azure.ai.ml import MLClient, command
        from azure.identity import DefaultAzureCredential
    except ImportError:
        print("❌ azure-ai-ml SDK'sı gereklidir")
        print("📦 Kurulum: pip install azure-ai-ml")
        return False
    
    # Config yükle
    config_path = Path(__file__).parent / "azure_ml_config.json"
    if not config_path.exists():
        print(f"❌ Config dosyası bulunamadı: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    SUBSCRIPTION_ID = config.get("subscription_id")
    RESOURCE_GROUP = config.get("resource_group")
    WORKSPACE_NAME = config.get("workspace_name")
    
    print(f"📋 Yapılandırma:")
    print(f"   • Workspace: {WORKSPACE_NAME}")
    print(f"   • Resource Group: {RESOURCE_GROUP}")
    print(f"   • Subscription: {SUBSCRIPTION_ID[:20]}...")
    print()
    
    try:
        # Credentials ve ML Client
        print("1️⃣  Azure credentials alınıyor...")
        credential = DefaultAzureCredential()
        
        print("2️⃣  ML Client bağlanıyor...")
        ml_client = MLClient(
            credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME
        )
        
        # Workspace bilgisi
        print("3️⃣  Workspace bilgisi alınıyor...")
        ws = ml_client.workspaces.get(name=WORKSPACE_NAME)
        print(f"   ✅ Workspace: {ws.name}")
        print(f"   ✅ Location: {ws.location}")
        
        # Job oluştur - Basit test
        print("4️⃣  Test job'u oluşturuluyor...")
        
        job = command(
            code=".",  # Mevcut dizin
            command="python run_l3_vfa_pma.py --input-data azureml://datastores/workspaceblobstore/paths/amos22-data/ --run_vfa_pma",
            environment="AzureML-sklearn-0.24",  # Default environment
            display_name="l3-vfa-pma-test-job",
            description="L3 VFA/PMA analysis test",
            compute="l3-compute-01"  # Compute instance adı
        )
        
        print("   ✅ Job tanımı oluşturuldu")
        
        # Job submit et
        print("5️⃣  Job Azure ML'e gönderiliyor...")
        returned_job = ml_client.jobs.create_or_update(job)
        
        print(f"\n✅ JOB BAŞARILI!")
        print(f"   • Job Name: {returned_job.name}")
        print(f"   • Job ID: {returned_job.id}")
        print(f"   • Status: {returned_job.status}")
        print(f"   • Compute: l3-compute-01")
        print()
        print(f"📊 Monitoring URL:")
        print(f"   https://ml.azure.com/jobs/{returned_job.name}?wsid=/subscriptions/{SUBSCRIPTION_ID}/resourcegroups/{RESOURCE_GROUP}/providers/microsoft.machinelearningservices/workspaces/{WORKSPACE_NAME}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Azure ML L3 VFA/PMA Pipeline Launcher")
    parser.add_argument("--run_vfa_pma", action="store_true", help="VFA/PMA işlemesini başlat")
    parser.add_argument("--quick-test", action="store_true", help="Hızlı test (CPU)")
    parser.add_argument("--wait", action="store_true", help="Job bitişini bekle")
    
    args = parser.parse_args()
    
    if args.run_vfa_pma or args.quick_test or len(sys.argv) == 1:
        success = launch_vfa_pma_job()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
