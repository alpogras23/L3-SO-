"""
TotalSegmentator Teacher Üretimi Utility

AMOS22 veya diğer CT NIfTI dataseti için TotalSegmentator ile
L3 seviyesi vertebra ve karın bölgesi maskelerini üretir.

Kullanım:
    python tools/generate_ts_teachers.py \
        --amos-root /content/drive/MyDrive/AMOS22 \
        --out-dir /content/drive/MyDrive/TS_teachers_AMOS22 \
        --tasks abdominal_muscles total \
        --fast
"""

import argparse
import subprocess
import sys
from pathlib import Path
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="TotalSegmentator Teacher Generator")
    parser.add_argument("--amos-root", type=str, required=True,
                        help="AMOS22 NIfTI kök dizini")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="TS çıktı kök dizini")
    parser.add_argument("--tasks", nargs="+", default=["abdominal_muscles", "total"],
                        help="TS task listesi")
    parser.add_argument("--fast", action="store_true",
                        help="--fast flag kullan (hızlı ama düşük çözünürlük)")
    parser.add_argument("--roi-subset", type=str, default="torso",
                        help="total task için ROI subset (örn: torso)")
    return parser.parse_args()


def find_nifti_files(root: Path):
    """AMOS22 NIfTI dosyalarını bul"""
    nifti_root = root / "imagesTr" if (root / "imagesTr").exists() else root
    files = sorted(list(nifti_root.glob("*.nii.gz")))
    if not files:
        files = sorted(list(nifti_root.rglob("*.nii.gz")))
    return files


def run_totalsegmentator(input_path: Path, output_dir: Path, task: str, fast: bool, roi_subset: str = None):
    """TotalSegmentator çalıştır (idempotent)"""
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / f"{task}_DONE"
    
    if marker.exists():
        return True  # Zaten yapılmış
    
    cmd = [
        sys.executable, "-m", "totalsegmentator",
        "-i", str(input_path),
        "-o", str(output_dir),
        "-ta", task,
    ]
    
    if fast:
        cmd.append("--fast")
    
    if task == "total" and roi_subset:
        cmd.extend(["--roi_subset", roi_subset])
    
    print(f"[RUN] {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        marker.touch()
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ TotalSegmentator hatası: {e}")
        return False


def main():
    args = parse_args()
    
    amos_root = Path(args.amos_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # NIfTI dosyalarını bul
    ct_files = find_nifti_files(amos_root)
    print(f"✅ Bulunan CT dosyası: {len(ct_files)}")
    
    if not ct_files:
        print("⚠️ NIfTI dosyası bulunamadı.")
        return
    
    # Her vaka için TS çalıştır
    success_count = 0
    for ct_path in tqdm(ct_files, desc="TotalSegmentator"):
        case_id = ct_path.stem.replace(".nii", "")
        case_out_dir = out_dir / case_id
        
        case_success = True
        for task in args.tasks:
            ok = run_totalsegmentator(
                ct_path, 
                case_out_dir, 
                task, 
                args.fast,
                args.roi_subset if task == "total" else None
            )
            if not ok:
                case_success = False
                break
        
        if case_success:
            success_count += 1
    
    print(f"\n✅ TotalSegmentator tamamlandı: {success_count}/{len(ct_files)} başarılı")
    print(f"📁 Çıktı dizini: {out_dir}")


if __name__ == "__main__":
    main()
