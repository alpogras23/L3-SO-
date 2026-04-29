#!/usr/bin/env python3
"""
Sadece C2C Teacher Mask Üretimi (VAT, SAT, psoas) - L3 Seviyesi
==============================================================

Masaüstü DICOMNET klasöründeki aksiyel BT kesitlerinden
Comp2Comp ile VAT, SAT, psoas teacher maskeleri üretir.

Kullanım:
    python scripts/generate_c2c_teachers_local.py

Çıktı:
    ~/Desktop/C2C_teachers_local/ klasöründe vaka bazında maskeler
    ve c2c_manifest.json
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import nibabel as nib
import pydicom
DEFAULT_DICOMNET_ROOT = Path.home() / "Desktop" / "DICOMNET"
DEFAULT_OUTPUT_ROOT = Path.home() / "Desktop" / "C2C_teachers_local"
DEFAULT_TEMP_NIFTI_DIR = Path.home() / "Desktop" / "temp_nifti_for_c2c"

def ensure_dirs(output_root: Path, temp_nifti_dir: Path):
    output_root.mkdir(exist_ok=True)
    temp_nifti_dir.mkdir(exist_ok=True)

def find_dicom_series(case_dir):
    """Bir vaka klasöründeki DICOM dosyalarını bul"""
    dicom_files = []
    for root, dirs, files in os.walk(case_dir):
        for f in files:
            # DICOM dosyası: .dcm, .DCM veya uzantısız
            if f.lower().endswith('.dcm') or ('.' not in f):
                dcm_path = Path(root) / f
                try:
                    pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
                    dicom_files.append(dcm_path)
                except Exception:
                    pass
    return sorted(dicom_files)

def dicom_to_nifti(dicom_files, output_nifti, force_fallback=False):
    """DICOM serilerini NIfTI'ye dönüştür.
    1) dcm2niix (varsa ve force_fallback=False)
    2) pydicom + nibabel custom stack
    """
    if len(dicom_files) == 0:
        return False

    if not force_fallback:
        dcm2niix_path = shutil.which("dcm2niix") or "/opt/homebrew/bin/dcm2niix"
        if dcm2niix_path and os.path.exists(dcm2niix_path):
            try:
                first_dicom_dir = dicom_files[0].parent
                cmd = [
                    dcm2niix_path,
                    "-f", output_nifti.stem,
                    "-o", str(output_nifti.parent),
                    "-z", "y",
                    "-m", "n",
                    str(first_dicom_dir)
                ]
                res = subprocess.run(cmd, capture_output=True, timeout=60)
                expected_output = output_nifti.parent / f"{output_nifti.stem}.nii.gz"
                if expected_output.exists():
                    if expected_output != output_nifti:
                        expected_output.rename(output_nifti)
                    return True
                else:
                    print("  ⚠️  dcm2niix çıktı bulunamadı (stderr):", res.stderr.decode("utf-8", errors="ignore")[:160])
            except Exception as e:
                print(f"  ⚠️  dcm2niix hatası: {e} -> fallback'e geçiliyor")
        else:
            print("  ℹ️  dcm2niix bulunamadı, fallback kullanılacak")
    else:
        print("  ℹ️  force_fallback=True, dcm2niix atlandı")

    # Fallback: pydicom slice stack
    try:
        slices = []
        positions = []
        for f in dicom_files:
            ds = pydicom.dcmread(str(f))
            pixel = ds.pixel_array.astype(np.int16)
            slices.append(pixel)
            if hasattr(ds, 'ImagePositionPatient'):
                positions.append(ds.ImagePositionPatient[2])
            else:
                positions.append(len(positions))
        order = np.argsort(positions)
        vol = np.stack([slices[i] for i in order], axis=-1)
        first = pydicom.dcmread(str(dicom_files[0]))
        try:
            px, py = map(float, first.PixelSpacing)
        except Exception:
            px, py = 1.0, 1.0
        try:
            sz = float(first.SliceThickness)
        except Exception:
            pos_sorted = np.sort(np.array(positions))
            sz = float(np.median(np.diff(pos_sorted))) if len(pos_sorted) > 1 else 1.0
        affine = np.diag([px, py, sz, 1.0])
        nib.save(nib.Nifti1Image(vol, affine), str(output_nifti))
        return True
    except Exception as e:
        print(f"  ❌ Fallback dönüşüm hatası: {e}")
        return False

def run_comp2comp(dicom_dir, output_dir, device="cpu"):
    """Comp2Comp CLI ile body composition inference (VAT, SAT, psoas)."""
    # C2C binary path
    c2c_bin = Path.home() / "Desktop" / "L3_SO_ANALYSIS" / "thirdparty" / "Comp2Comp" / "bin" / "C2C"
    if not c2c_bin.exists():
        print(f"    ❌ C2C binary bulunamadı: {c2c_bin}")
        return False
    
    # .c2c_env python interpreter
    c2c_python = Path.home() / "Desktop" / "L3_SO_ANALYSIS" / ".c2c_env" / "bin" / "python"
    if not c2c_python.exists():
        print(f"    ❌ .c2c_env python bulunamadı: {c2c_python}")
        return False
    
    try:
        # C2C muscle_adipose_tissue pipeline çalıştır (.c2c_env ile DICOM input)
        cmd = [
            str(c2c_python),
            str(c2c_bin),
            "muscle_adipose_tissue",
            "--input_path", str(dicom_dir),
            "--output_path", str(output_dir),
            "--save_segmentations",
            "--muscle_fat_model", "abCT_v0.0.1"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=600,  # 10 dakika (CPU inference yavaş)
            text=True
        )
        
        if result.returncode != 0:
            print(f"    ⚠️  C2C çıkış kodu: {result.returncode}")
            print(f"    stderr: {result.stderr[:400]}")
            return False
        
        # Beklenen çıktılar (C2C çıktı yapısına göre)
        # abCT_v0.0.1 modeli medical_volume.nii.gz + segmentations/ klasörü oluşturur
        seg_dir = output_dir / "segmentations"
        if not seg_dir.exists():
            print(f"    ⚠️  Segmentasyon klasörü bulunamadı: {seg_dir}")
            return False
        
        # VAT, SAT, psoas maskelerini kontrol et veya dosyaları eşle
        # C2C çıktı formatı: multichannel segmentation veya ayrı dosyalar
        # Basitleştirilmiş kontrol: segmentation dosyası varsa başarılı
        seg_files = list(seg_dir.glob("*.nii.gz"))
        if len(seg_files) == 0:
            print(f"    ⚠️  Segmentasyon çıktısı bulunamadı")
            return False
        
        print(f"    ✅ {len(seg_files)} segmentasyon dosyası oluşturuldu")
        return True
        
    except subprocess.TimeoutExpired:
        print("    ❌ C2C timeout (300s)")
        return False
    except Exception as e:
        print(f"    ❌ C2C çalışma hatası: {e}")
        import traceback
        traceback.print_exc(limit=2)
        return False

def parse_args():
    ap = argparse.ArgumentParser(description="Sadece C2C Teacher Mask Üretimi (VAT, SAT, psoas)")
    ap.add_argument("--dicom-root", type=Path, default=DEFAULT_DICOMNET_ROOT, help="DICOM vaka kökü")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Çıktı kökü")
    ap.add_argument("--temp-root", type=Path, default=DEFAULT_TEMP_NIFTI_DIR, help="Geçici NIfTI dizini")
    ap.add_argument("--limit", type=int, default=None, help="Maksimum vaka sayısı (hızlı test için)")
    ap.add_argument("--force-fallback", action="store_true", help="dcm2niix'i atla, direkt pydicom fallback kullan")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Comp2Comp cihazı")
    return ap.parse_args()

def main():
    args = parse_args()
    dicom_root = args.dicom_root
    output_root = args.out_root
    temp_root = args.temp_root
    ensure_dirs(output_root, temp_root)

    print("=" * 70)
    print("🔬 Sadece C2C Teacher Mask Üretimi (VAT, SAT, psoas) - L3 Seviyesi")
    print("=" * 70)
    print(f"\n📂 DICOM Root: {dicom_root}")
    print(f"📂 Output Root: {output_root}")
    print(f"📂 Temp NIfTI: {temp_root}\n")

    case_dirs = [d for d in dicom_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if args.limit:
        case_dirs = case_dirs[:args.limit]
    print(f"✅ {len(case_dirs)} vaka işlenecek (limit={args.limit})\n")

    success_count = 0
    error_count = 0

    for case_dir in tqdm(case_dirs, desc="C2C Teacher Üretimi"):
        case_id = case_dir.name
        output_case_dir = output_root / case_id

        if output_case_dir.exists() and {"vat.nii.gz", "sat.nii.gz", "psoas.nii.gz"}.issubset({p.name for p in output_case_dir.iterdir()} if output_case_dir.exists() else set()):
            success_count += 1
            continue

        output_case_dir.mkdir(exist_ok=True)

        dicom_files = find_dicom_series(case_dir)
        if len(dicom_files) == 0:
            print(f"\n  ⚠️  {case_id}: DICOM bulunamadı")
            error_count += 1
            continue

        # C2C DICOM klasörünü doğrudan kullanır (DicomFinder)
        if run_comp2comp(case_dir, output_case_dir, device=args.device):
            success_count += 1
        else:
            error_count += 1

    print("\n" + "=" * 70)
    print("✅ Sadece C2C Teacher Üretimi Tamamlandı!")
    print("=" * 70)
    print(f"  Başarılı: {success_count}/{len(case_dirs)}")
    print(f"  Hatalı: {error_count}/{len(case_dirs)}")
    print(f"\n📂 Çıktılar: {output_root}")

    manifest_data = []
    for case_dir in output_root.iterdir():
        if case_dir.is_dir():
            manifest_data.append({
                "case_id": case_dir.name,
                "c2c_output_dir": str(case_dir),
                "vat_mask": str(case_dir / "vat.nii.gz") if (case_dir / "vat.nii.gz").exists() else None,
                "sat_mask": str(case_dir / "sat.nii.gz") if (case_dir / "sat.nii.gz").exists() else None,
                "psoas_mask": str(case_dir / "psoas.nii.gz") if (case_dir / "psoas.nii.gz").exists() else None
            })

    manifest_path = output_root / "c2c_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\n📄 Manifest: {manifest_path}")

if __name__ == "__main__":
    # Paket kontrolü (pydicom ve nibabel üst kısımda zaten import edildi)
    try:
        import pydicom  # noqa: F401
        import nibabel  # noqa: F401
    except ImportError as e:
        print("❌ Gerekli paket eksik:", e)
        print("   pip install pydicom nibabel")
        sys.exit(1)

    # dcm2niix kontrolü (opsiyonel uyarı)
    try:
        subprocess.run(["dcm2niix", "-h"], capture_output=True, timeout=5)
    except Exception:
        print("⚠️  dcm2niix bulunamadı. Kurulum:")
        print("   brew install dcm2niix  # macOS")
        print("   Veya: https://github.com/rordenlab/dcm2niix")

    main()
