#!/usr/bin/env python3
"""
TotalSegmentator Teacher Mask Üretimi - C2C NIfTI Dosyalarından
=================================================================

C2C için oluşturulan NIfTI dosyalarını kullanarak
TotalSegmentator ile vertebra, kas ve body mask üretir.

Kullanım:
    python scripts/generate_ts_teachers_from_c2c.py

Çıktı:
    ~/Desktop/TS_teachers_local/ klasöründe vaka bazında maskeler
"""

import os
import sys
from pathlib import Path
import subprocess
from tqdm import tqdm
import json

# NIfTI dosya yolu (C2C işleminden sonra)
C2C_NIFTI_DIR = Path.home() / "Desktop" / "temp_nifti_for_c2c"
OUTPUT_ROOT = Path.home() / "Desktop" / "TS_teachers_local"

# Alternatif: DICOMNET'ten direkt NIfTI oluştur
DICOMNET_ROOT = Path.home() / "Desktop" / "DICOMNET"

OUTPUT_ROOT.mkdir(exist_ok=True)

def run_totalsegmentator(nifti_path, output_dir):
    """TotalSegmentator ile segmentasyon yap"""
    try:
        cmd = [
            "TotalSegmentator",
            "-i", str(nifti_path),
            "-o", str(output_dir),
            "--fast",
            "--ml"
        ]
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=300,
            text=True
        )
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n    ❌ TotalSegmentator hatası:")
        print(f"       {e.stderr[:200]}")
        return False
    except subprocess.TimeoutExpired:
        print(f"\n    ⏱️  Timeout (>5 dk)")
        return False
    except Exception as e:
        print(f"\n    ❌ Beklenmeyen hata: {e}")
        return False

def find_nifti_files(search_dir):
    """Bir klasördeki NIfTI dosyalarını bul"""
    nifti_files = []
    for f in search_dir.rglob("*.nii.gz"):
        if f.is_file():
            nifti_files.append(f)
    return sorted(nifti_files)

def main():
    print("=" * 70)
    print("🔬 TotalSegmentator Teacher Mask Üretimi")
    print("=" * 70)
    print(f"\n📂 NIfTI Source: {C2C_NIFTI_DIR}")
    print(f"📂 Output Root: {OUTPUT_ROOT}\n")
    
    # NIfTI dosyalarını bul
    nifti_files = find_nifti_files(C2C_NIFTI_DIR)
    
    if len(nifti_files) == 0:
        print("⚠️  NIfTI dosyası bulunamadı!")
        print(f"Lütfen önce C2C teacher script'ini çalıştırın.\n")
        print("Alternatif: DICOM → NIfTI dönüşümü yapılsın mı? (E/H)")
        response = input().strip().upper()
        
        if response == "E":
            print("\nDICOM → NIfTI dönüşümü başlatılıyor...")
            # TODO: DICOM → NIfTI batch conversion
            sys.exit(0)
        else:
            sys.exit(1)
    
    print(f"✅ {len(nifti_files)} NIfTI dosyası bulundu\n")
    
    success_count = 0
    error_count = 0
    
    for nifti_path in tqdm(nifti_files, desc="TotalSegmentator"):
        case_id = nifti_path.stem.replace(".nii", "")
        output_case_dir = OUTPUT_ROOT / case_id
        
        # Zaten işlenmişse atla
        if output_case_dir.exists() and len(list(output_case_dir.iterdir())) > 0:
            success_count += 1
            continue
        
        output_case_dir.mkdir(exist_ok=True)
        
        # TotalSegmentator çalıştır
        if run_totalsegmentator(nifti_path, output_case_dir):
            success_count += 1
        else:
            error_count += 1
    
    print("\n" + "=" * 70)
    print("✅ TotalSegmentator Teacher Üretimi Tamamlandı!")
    print("=" * 70)
    print(f"  Başarılı: {success_count}/{len(nifti_files)}")
    print(f"  Hatalı: {error_count}/{len(nifti_files)}")
    print(f"\n📂 Çıktılar: {OUTPUT_ROOT}")
    
    # Manifest oluştur
    manifest_data = []
    for case_dir in OUTPUT_ROOT.iterdir():
        if case_dir.is_dir():
            mask_files = list(case_dir.glob("*.nii.gz"))
            manifest_data.append({
                "case_id": case_dir.name,
                "ts_output_dir": str(case_dir),
                "mask_count": len(mask_files),
                "mask_files": [f.name for f in mask_files[:10]]  # İlk 10
            })
    
    manifest_path = OUTPUT_ROOT / "ts_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    
    print(f"\n📄 Manifest: {manifest_path}")

if __name__ == "__main__":
    # TotalSegmentator kontrolü
    try:
        subprocess.run(["TotalSegmentator", "--help"], capture_output=True, timeout=5)
        print("✅ TotalSegmentator kurulu\n")
    except Exception:
        print("❌ TotalSegmentator bulunamadı!")
        print("Kurulum:")
        print("   pip install TotalSegmentator>=2.3.0\n")
        sys.exit(1)
    
    main()
