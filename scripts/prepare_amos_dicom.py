import argparse
import subprocess
from pathlib import Path

try:
    import SimpleITK as sitk  # plastimatch yoksa fallback
    _HAS_SITK = True
except Exception:
    _HAS_SITK = False

"""
AMOS NIfTI -> DICOM dönüştürücü (plastimatch gerektirir)

Kullanım örneği:
python scripts/prepare_amos_dicom.py --amos-root \
  /Users/you/Desktop/AMOS22 --out-root data/amos_dicom

- Her .nii.gz CT volümü için bir DICOM seri klasörü üretir
- plastimatch yoksa kurulum önerisi verir (macOS: brew install plastimatch)
"""


def check_plastimatch() -> bool:
    try:
        out = subprocess.run(["plastimatch", "--version"], capture_output=True, text=True)
        return out.returncode == 0
    except FileNotFoundError:
        return False


def convert_one(nifti_path: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "plastimatch",
        "convert",
        "--input", str(nifti_path),
        "--output-dicom", str(out_dir),
    ]
    print("[AMOS->DICOM]", " ".join(cmd))
    res = subprocess.run(cmd)
    return res.returncode


def convert_one_sitk(nifti_path: Path, out_dir: Path) -> int:
    """Basit SimpleITK fallback: NIfTI hacmi dilimlere ayırıp temel DICOM dosyaları yazar.
    Not: Klinik-grade tag setleri için plastimatch önerilir.
    """
    if not _HAS_SITK:
        print("[AMOS->DICOM][fallback] SimpleITK bulunamadı. Lütfen 'pip install SimpleITK' kurun.")
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        img = sitk.ReadImage(str(nifti_path))
        size = img.GetSize()  # (x,y,z)
        # Basit Int16’a yeniden ölçekleme (CT HU benzeri aralık varsayımı yok; sadece demo amaçlı)
        img16 = sitk.Cast(sitk.RescaleIntensity(img, outputMinimum=-1024, outputMaximum=3071), sitk.sitkInt16)
        for z in range(size[2]):
            sl = img16[:, :, z]
            # Minimal zorunlu tag’ler
            sl = sitk.JoinSeries(sl)  # 2D -> 3D tek dilim şeklinde
            # Tek dilim yaz: .dcm
            out_file = out_dir / f"slice_{z:04d}.dcm"
            sitk.WriteImage(sl, str(out_file), True)
        print(f"[AMOS->DICOM][fallback] Wrote {size[2]} slices to {out_dir}")
        return 0
    except Exception as e:
        print("[AMOS->DICOM][fallback] SimpleITK yazım hatası:", e)
        return 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amos-root", required=True, help="AMOS 22 dataset kök dizini (NIfTI)")
    ap.add_argument("--out-root", required=True, help="Çıktı DICOM kökü")
    args = ap.parse_args()

    has_pm = check_plastimatch()
    if not has_pm and not _HAS_SITK:
        print("[WARN] plastimatch ve SimpleITK yok. Seçenekler:\n- brew install plastimatch (macOS)\n- pip install SimpleITK (fallback)")
        return 2

    amos_root = Path(args.amos_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    nii_list = list(amos_root.rglob("*.nii.gz"))
    if not nii_list:
        print("[AMOS->DICOM] .nii.gz bulunamadı:", amos_root)
        return 1

    ok = fail = 0
    for nii in nii_list:
        rel = nii.relative_to(amos_root)
        series_dir = out_root / rel.with_suffix("").with_suffix("")  # .nii.gz -> stem
        if has_pm:
            rc = convert_one(nii, series_dir)
        else:
            rc = convert_one_sitk(nii, series_dir)
        if rc == 0:
            ok += 1
        else:
            fail += 1
    print(f"[AMOS->DICOM] tamam: {ok}, hata: {fail}")
    return 0 if fail == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
