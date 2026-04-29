#!/usr/bin/env python3
"""
AMOS22 datasını Azure Blob Storage'a yükle
"""

import json
from pathlib import Path
import subprocess
import sys

def upload_to_azure_blob():
    """Azure Storage'a yükle"""
    
    print("\n" + "="*70)
    print("📤 AMOS22 DATASINI AZURE STORAGE'A YÜKLEME")
    print("="*70 + "\n")
    
    # Config yükle
    config_path = Path(__file__).parent / "azure_ml_config.json"
    if not config_path.exists():
        print(f"❌ Config dosyası bulunamadı: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    STORAGE_ACCOUNT = config.get("storage_account")
    STORAGE_KEY = config.get("storage_connection_string", "").split("AccountKey=")[1].split(";")[0] if "AccountKey=" in config.get("storage_connection_string", "") else None
    CONTAINER = config.get("storage_container")
    
    if not STORAGE_ACCOUNT or not STORAGE_KEY:
        print(f"❌ Storage bilgisi config'den alınamadı")
        print(f"   Storage Account: {STORAGE_ACCOUNT}")
        print(f"   Storage Key: {'Var' if STORAGE_KEY else 'YOK'}")
        return False
    
    # Yüklenecek veri
    UPLOAD_DIR = Path(__file__).parent.parent.parent / "azure_upload_test"
    
    if not UPLOAD_DIR.exists():
        print(f"❌ Upload dizini bulunamadı: {UPLOAD_DIR}")
        print(f"   Lütfen önce prepare_amos_azure.py çalıştırın")
        return False
    
    print(f"📊 Yapılandırma:")
    print(f"   • Storage Account: {STORAGE_ACCOUNT}")
    print(f"   • Container: {CONTAINER}")
    print(f"   • Yüklenecek Dizin: {UPLOAD_DIR}")
    print(f"   • Toplam Dosya Sayısı: {len(list(UPLOAD_DIR.glob('*')))} case")
    print()
    
    try:
        # Azure CLI ile yükle
        print("1️⃣  Azure Storage'a bağlanılıyor...")
        
        # Ortam değişkenleri ayarla
        import os
        os.environ['AZURE_STORAGE_ACCOUNT'] = STORAGE_ACCOUNT
        os.environ['AZURE_STORAGE_KEY'] = STORAGE_KEY
        
        # Upload-batch komutu
        cmd = [
            "az", "storage", "blob", "upload-batch",
            "--account-name", STORAGE_ACCOUNT,
            "--account-key", STORAGE_KEY,
            "--destination", CONTAINER,
            "--source", str(UPLOAD_DIR),
            "--pattern", "*",
            "--verbose"
        ]
        
        print("2️⃣  Dosyalar yükleniyor...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            print(result.stdout)
            print("\n✅ YÜKLEME BAŞARILI!")
            
            # Dosya sayısını kontrol et
            check_cmd = [
                "az", "storage", "blob", "list",
                "--account-name", STORAGE_ACCOUNT,
                "--account-key", STORAGE_KEY,
                "--container-name", CONTAINER,
                "--query", "length(@)"
            ]
            
            count_result = subprocess.run(check_cmd, capture_output=True, text=True)
            blob_count = count_result.stdout.strip()
            
            print(f"\n📊 Azure Storage Durumu:")
            print(f"   • Toplam Blob: {blob_count}")
            print(f"   • Container: {CONTAINER}")
            print(f"   • Storage Account: {STORAGE_ACCOUNT}")
            
            return True
        else:
            print(f"❌ Yükleme Hatası:")
            print(result.stderr)
            return False
    
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = upload_to_azure_blob()
    sys.exit(0 if success else 1)
