import argparse
import os
import shutil
import subprocess
from pathlib import Path

"""
DICOM serileri üzerinde öğretmen çıkarımları (TS ve C2C) çalıştırır.
- TS: TotalSegmentator (opsiyonel)
- C2C: harici bir CLI ya da python modülü (kullanıcı taraflı)

Bu betik sadece orkestrasyon sağlar; çıktıların NIfTI veya PNG olarak nerede
oluştuğunu bildirir. Ayrıntılı post-process bu sürümde yapılmaz.
"""


def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run_ts(series_dir: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not which("TotalSegmentator"):
        print("[TS] TotalSegmentator bulunamadı. pip install TotalSegmentator && nnUNetv2 gerektirir.")
        return 2
    cmd = [
        "TotalSegmentator",
        "-i",
        str(series_dir),
        "-o",
        str(out_dir),
        "--fast",
    ]
    print("[TS]", " ".join(cmd))
    return subprocess.run(cmd).returncode


def run_c2c(series_dir: Path, out_dir: Path, c2c_cmd: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not which(c2c_cmd):
        print(f"[C2C] {c2c_cmd} bulunamadı. Lütfen C2C öğretmen CLI/komutunu kurun.")
        return 2
    cmd = [c2c_cmd, str(series_dir), str(out_dir)]
    print("[C2C]", " ".join(cmd))
    return subprocess.run(cmd).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-root", required=True, help="DICOM serileri kökü (prepare_amos_dicom çıktısı)")
    ap.add_argument("--out-root", required=True, help="Öğretmen çıktı kökü")
    ap.add_argument("--c2c-cmd", default=os.environ.get("C2C_CLI", "c2c_infer"))
    args = ap.parse_args()

    dicom_root = Path(args.dicom_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    series = [p for p in dicom_root.glob("**/*") if p.is_dir() and any(p.iterdir())]
    if not series:
        print("[TEACHERS] DICOM seri klasörü bulunamadı:", dicom_root)
        return 1

    for s in series:
        rel = s.relative_to(dicom_root)
        ts_out = out_root / "ts" / rel
        c2c_out = out_root / "c2c" / rel
        run_ts(s, ts_out)
        run_c2c(s, c2c_out, args.c2c_cmd)

    print("[TEACHERS] tamamlandı.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
