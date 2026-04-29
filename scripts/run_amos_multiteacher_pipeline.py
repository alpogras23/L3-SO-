import argparse
import subprocess
from pathlib import Path
import json
from typing import List, Tuple

import numpy as np
import SimpleITK as sitk

"""
AMOS -> DICOM -> (Öğretmenler) -> L3 PNG export -> Manifest -> (opsiyonel) Eğitim
Tek komutta uçtan uca pipeline.

Örnek:
python scripts/run_amos_multiteacher_pipeline.py \
  --amos-root "$HOME/Desktop/AMOS22" \
  --out-root data/pipeline \
  --images-out psoas_ml/data/images \
  --gt-out psoas_ml/data/gt_masks \
  --c2c-out psoas_ml/data/c2c_masks \
  --c2c-l3-id 55 --c2c-psoas-ids 55,56 --c2c-vat-id 100 --c2c-sat-id 101 \
  --gt-psoas-ids 55 \
  --gt-labels-root data/amos_labels \
  --gt-label-pattern "{stem}_gt.nii.gz" \
  --c2c-labels-root data/teachers/c2c \
  --c2c-label-pattern "{stem}/labels.nii.gz" \
  --weights 1.0,0.3,0.3 \
  --epochs 60 --threads 1 --throttle-ms 10 --train

Not: Öğretmenleri çalıştırmak bu scriptin içinde opsiyoneldir; TotalSegmentator/C2C
kurulu değilse ilgili adım atlanır ve mevcut label haritaları kullanılır.
"""

ROOT = Path(__file__).resolve().parents[1]
PY = str((ROOT / ".venv" / "bin" / "python").resolve())
PY_FALLBACK = "python3"


def _run(cmd):
    print("[PIPE] $", " ".join(cmd))
    return subprocess.run(cmd).returncode


def which(cmd: str) -> bool:
    from shutil import which as _which
    return _which(cmd) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amos-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--images-out", default=str(ROOT / "psoas_ml/data/images"))
    ap.add_argument("--gt-out", default=str(ROOT / "psoas_ml/data/gt_masks"))
    ap.add_argument("--c2c-out", default=str(ROOT / "psoas_ml/data/c2c_masks"))
    ap.add_argument("--ts-out", default=str(ROOT / "psoas_ml/data/ts_masks"))

    ap.add_argument("--gt-labels-root", default="")
    ap.add_argument("--gt-label-pattern", default="{stem}_gt.nii.gz")
    ap.add_argument("--c2c-labels-root", default="")
    ap.add_argument("--c2c-label-pattern", default="{stem}/labels.nii.gz")
    ap.add_argument("--ts-labels-root", default="")
    ap.add_argument("--ts-label-pattern", default="{stem}/labels.nii.gz")
    ap.add_argument("--labels-config", default="", help="JSON preset for IDs & weights")

    # TS per-class otomatik birleştirme (opsiyonel):
    #   ts-perclass-root/{stem}/ altında per-class NIfTI'ler varsa bunları tek çok-etiketli NIfTI'ye birleştirir
    #   ve export aşamasında TS label map olarak kullanır.
    #   Örnek: --ts-perclass-root data/ts_perclass --ts-perclass-out-root data/ts_merged \
    #          --ts-merge-label 55:psoas_left*.nii.gz --ts-merge-label 56:psoas_right*.nii.gz
    ap.add_argument("--ts-perclass-root", default="", help="TS per-class kök dizini (case klasörleri {stem} adına göre)")
    ap.add_argument("--ts-perclass-out-root", default="", help="Birleştirilmiş TS çıktı kökü (varsayılan: out-root/ts_merged)")
    ap.add_argument("--ts-merge-label", action="append", default=[], help="ID:glob (case klasörü içinde eşleşecek pattern)")
    ap.add_argument("--ts-merge-reference", default="", help="(opsiyonel) referans NIfTI göre geometriyi sabitler; case içinde relative glob")

    ap.add_argument("--case-glob", default="**/*.nii.gz")
    ap.add_argument("--prefix-from-stem", action="store_true")

    # Label id/ids
    ap.add_argument("--c2c-l3-id", type=int, default=-1)
    ap.add_argument("--c2c-vat-id", type=int, default=-1)
    ap.add_argument("--c2c-sat-id", type=int, default=-1)
    ap.add_argument("--c2c-psoas-ids", default="")
    ap.add_argument("--gt-psoas-ids", default="")
    ap.add_argument("--ts-psoas-ids", default="")

    ap.add_argument("--weights", default="1.0,0.3,0.3", help="w_gt,w_ts,w_c2c")
    ap.add_argument("--manifest", default=str(ROOT / "psoas_ml/data/multiteacher_manifest.csv"))

    # Öğretmen çalıştırma
    ap.add_argument("--run-teachers", action="store_true")

    # Eğitim opsiyonları
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--throttle-ms", type=int, default=10)
    ap.add_argument("--min-usable", type=int, default=1, help="train'i başlatmak için en az bu kadar usable satır olsun")

    args = ap.parse_args()

    # Load labels config if provided
    if args.labels_config:
        try:
            with open(args.labels_config, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
        except Exception as e:
            print("[PIPE] WARN: labels-config okunamadı:", e)
            cfg = {}
        # Apply defaults from config only if CLI values are not set
        c2c_cfg = cfg.get("c2c", {})
        gt_cfg = cfg.get("gt", {})
        ts_cfg = cfg.get("ts", {})
        w_cfg = cfg.get("weights", {})
        if args.c2c_l3_id < 0 and "l3_id" in c2c_cfg:
            args.c2c_l3_id = int(c2c_cfg["l3_id"])
        if not args.c2c_psoas_ids and "psoas_ids" in c2c_cfg:
            args.c2c_psoas_ids = ",".join(str(x) for x in c2c_cfg["psoas_ids"])  # comma-separated
        if args.c2c_vat_id < 0 and "vat_id" in c2c_cfg:
            args.c2c_vat_id = int(c2c_cfg["vat_id"])
        if args.c2c_sat_id < 0 and "sat_id" in c2c_cfg:
            args.c2c_sat_id = int(c2c_cfg["sat_id"])
        if not args.gt_psoas_ids and "psoas_ids" in gt_cfg:
            args.gt_psoas_ids = ",".join(str(x) for x in gt_cfg["psoas_ids"])  # comma-separated
        if not args.ts_psoas_ids and "psoas_ids" in ts_cfg:
            args.ts_psoas_ids = ",".join(str(x) for x in ts_cfg["psoas_ids"])  # comma-separated
        if args.weights == "1.0,0.3,0.3" and w_cfg:
            args.weights = f"{w_cfg.get('w_gt',1.0)},{w_cfg.get('w_ts',0.3)},{w_cfg.get('w_c2c',0.3)}"

    amos_root = Path(args.amos_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    dicom_root = out_root / "amos_dicom"
    teachers_root = out_root / "teachers"

    images_out = Path(args.images_out)
    images_out.mkdir(parents=True, exist_ok=True)
    gt_out = Path(args.gt_out)
    gt_out.mkdir(parents=True, exist_ok=True)
    c2c_out = Path(args.c2c_out)
    c2c_out.mkdir(parents=True, exist_ok=True)
    ts_out = Path(args.ts_out)
    ts_out.mkdir(parents=True, exist_ok=True)

    gt_labels_root = Path(args.gt_labels_root) if args.gt_labels_root else None
    c2c_labels_root = Path(args.c2c_labels_root) if args.c2c_labels_root else None
    ts_labels_root = Path(args.ts_labels_root) if args.ts_labels_root else None

    # TS per-class merge çıktı kökü
    ts_merged_root = Path(args.ts_perclass_out_root) if args.ts_perclass_out_root else (out_root / "ts_merged")
    if args.ts_perclass_root:
        ts_merged_root.mkdir(parents=True, exist_ok=True)

    # Hazırla: ts-merge-label -> List[Tuple[int, str]]
    ts_merge_specs: List[Tuple[int, str]] = []
    for spec in args.ts_merge_label:
        try:
            sid, sglob = spec.split(":", 1)
            ts_merge_specs.append((int(sid), sglob))
        except Exception:
            print("[PIPE] WARN: Geçersiz --ts-merge-label:", spec)

    def merge_ts_perclass(case_stem: str) -> Path:
        """Per-class TS NIfTI'leri tek çok-etiketli haritaya birleştirir ve çıktının yolunu döndürür.
        Varsayım: case klasörü ts-perclass-root/{stem}/ şeklinde.
        """
        case_dir = Path(args.ts_perclass_root) / case_stem
        if not case_dir.exists():
            print("[PIPE] WARN: TS per-class case dizini yok:", case_dir)
            return Path()

        # Referans görüntü belirle: seçenek 1) kullanıcı pattern'i 2) ilk bulunan label
        ref_img = None
        ref_path = None
        if args.ts_merge_reference:
            ref_matches = list(case_dir.glob(args.ts_merge_reference))
            if ref_matches:
                ref_path = ref_matches[0]
                ref_img = sitk.ReadImage(str(ref_path))

        # Etiketleri sırayla uygula (son kazansın)
        out_arr = None
        for lab_id, g in ts_merge_specs:
            matches = list(case_dir.glob(g))
            if not matches:
                print(f"[PIPE] WARN: TS per-class eşleşme yok ({lab_id}:{g}) case={case_stem}")
                continue
            mpath = matches[0]
            img = sitk.ReadImage(str(mpath))
            arr = sitk.GetArrayFromImage(img)
            if ref_img is None:
                ref_img = img
                ref_path = mpath
            if out_arr is None:
                out_arr = np.zeros_like(arr, dtype=np.int16)
            mask = (arr > 0)
            out_arr[mask] = int(lab_id)

        if ref_img is None or out_arr is None:
            print("[PIPE] WARN: TS per-class birleştirme için yeterli girdi yok, atlanıyor:", case_stem)
            return Path()

        out_img = sitk.GetImageFromArray(out_arr)
        out_img.SetSpacing(ref_img.GetSpacing())
        out_img.SetDirection(ref_img.GetDirection())
        out_img.SetOrigin(ref_img.GetOrigin())

        out_case_dir = ts_merged_root / case_stem
        out_case_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_case_dir / "labels.nii.gz"
        sitk.WriteImage(out_img, str(out_path))
        print("[PIPE] [TS-MERGE] Yazıldı:", out_path)
        return out_path

    # 1) AMOS -> DICOM
    rc = _run([PY if Path(PY).exists() else PY_FALLBACK,
               str(ROOT / "scripts/prepare_amos_dicom.py"),
               "--amos-root", str(amos_root),
               "--out-root", str(dicom_root)])
    if rc != 0:
        print("[PIPE] WARN: AMOS->DICOM dönüşümünde hata (fallback vs. kontrol edin). Devam ediliyor.")

    # 2) Öğretmenler (opsiyonel)
    if args.run_teachers:
        rc = _run([PY if Path(PY).exists() else PY_FALLBACK,
                   str(ROOT / "scripts/run_teachers.py"),
                   "--dicom-root", str(dicom_root),
                   "--out-root", str(teachers_root)])
        if rc != 0:
            print("[PIPE] WARN: Öğretmen çalıştırma çıkış kodu:", rc)

    # 3) Her vaka için L3 export
    vol_list = sorted(amos_root.glob(args.case_glob))
    if not vol_list:
        print("[PIPE] HATA: AMOS kökünde NIfTI bulunamadı:", amos_root)
        return 2

    exported = 0
    for nii in vol_list:
        stem = nii.stem
        if stem.endswith('.nii'):
            stem = stem[:-4]
        prefix = stem if args.prefix_from_stem else stem

        # DICOM seri klasörü (prepare_amos_dicom.py'nin çıktısına göre)
        series_dir = dicom_root / nii.relative_to(amos_root).with_suffix("").with_suffix("")
        if not series_dir.exists():
            print("[PIPE] WARN: DICOM seri bulunamadı, atlanıyor:", series_dir)
            continue

        # C2C ve GT label map’leri
        c2c_map = None
        if c2c_labels_root:
            c2c_map = c2c_labels_root / args.c2c_label_pattern.format(stem=stem)
        gt_map = None
        if gt_labels_root:
            gt_map = gt_labels_root / args.gt_label_pattern.format(stem=stem)
        ts_map = None
        if ts_labels_root:
            ts_map = ts_labels_root / args.ts_label_pattern.format(stem=stem)
        # TS per-class otomatik birleştirme devredeyse ve ts_map yoksa birleştirme dene
        if (not ts_map or not ts_map.exists()) and args.ts_perclass_root and ts_merge_specs:
            merged = merge_ts_perclass(stem)
            if merged and merged.exists():
                ts_map = merged

        cmd = [PY if Path(PY).exists() else PY_FALLBACK,
               str(ROOT / "scripts/export_l3_pngs.py"),
               "--dicom-series", str(series_dir),
               "--out-images", str(images_out),
               "--out-c2c", str(c2c_out),
               "--prefix", prefix]
        if c2c_map and c2c_map.exists():
            cmd += ["--c2c-label-map", str(c2c_map),
                    "--label-c2c-l3", str(args.c2c_l3_id),
                    "--label-c2c-vat", str(args.c2c_vat_id),
                    "--label-c2c-sat", str(args.c2c_sat_id),
                    "--label-c2c-psoas", str(args.c2c_psoas_ids),
                    "--c2c-base-from-psoas"]
        if gt_map and gt_map.exists():
            cmd += ["--out-gt", str(gt_out),
                    "--gt-label-map", str(gt_map),
                    "--label-gt-psoas", str(args.gt_psoas_ids)]
        if ts_map and ts_map.exists():
            cmd += ["--out-ts", str(ts_out),
                    "--ts-label-map", str(ts_map),
                    "--label-ts-psoas", str(args.ts_psoas_ids)]

        rc = _run(cmd)
        if rc == 0:
            exported += 1
        else:
            print("[PIPE] WARN: export_l3_pngs çıktı kodu:", rc, "vaka:", stem)

    print(f"[PIPE] L3 export tamamlanan vaka sayısı: {exported}/{len(vol_list)}")

    # 4) Manifest
    w_gt, w_ts, w_c2c = (args.weights.split(",") + ["1.0","0.3","0.3"])[:3]
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    rc = _run([PY if Path(PY).exists() else PY_FALLBACK,
               str(ROOT / "scripts/build_multiteacher_manifest.py"),
               "--images-root", str(images_out),
               "--gt-root", str(gt_out),
               "--ts-root", str(ts_out),
               "--c2c-root", str(c2c_out),
               "--out", str(manifest),
               "--w-gt", str(w_gt), "--w-ts", str(w_ts), "--w-c2c", str(w_c2c)])
    if rc != 0:
        print("[PIPE] WARN: manifest oluşturma hatası")

    # Manifest kullanılabilirlik kontrolü
    usable = 0
    try:
        import csv
        with open(manifest, "r", encoding="utf-8") as fh:
            r = csv.DictReader(fh)
            for row in r:
                has_gt = bool(row.get("gt_mask"))
                has_ts = bool(row.get("ts_mask"))
                has_c2c = bool(row.get("c2c_mask"))
                if has_gt or has_ts or has_c2c:
                    usable += 1
        print(f"[PIPE] Manifest usable satır sayısı: {usable}")
    except Exception as e:
        print("[PIPE] WARN: manifest okunamadı:", e)

    # 5) Eğitim (opsiyonel)
    if args.train:
        if usable < int(args.min_usable):
            print(f"[PIPE] INFO: usable < {args.min_usable}; eğitim başlatılmayacak.")
            return 0
        rc = _run([PY if Path(PY).exists() else PY_FALLBACK,
                   str(ROOT / "psoas_ml/multiteacher_train.py"),
                   "--manifest", str(manifest),
                   "--epochs", str(args.epochs),
                   "--threads", str(args.threads),
                   "--throttle-ms", str(args.throttle_ms),
                   "--out", str(ROOT / "psoas_ml/ckpts")])
        if rc != 0:
            print("[PIPE] WARN: eğitim çıkış kodu:", rc)

    print("[PIPE] Bitti.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
