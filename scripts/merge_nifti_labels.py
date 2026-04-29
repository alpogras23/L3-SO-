import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import SimpleITK as sitk

"""
Per-class NIfTI dosyalarını tek multi-label NIfTI'ye birleştirir.

Kullanım örnekleri:

1) Sıra ile öncelik vererek (ilk giren en düşük öncelik, son giren en yüksek):
python scripts/merge_nifti_labels.py \
  --out merged_labels.nii.gz \
  --label 55:/path/psoas_left.nii.gz \
  --label 56:/path/psoas_right.nii.gz \
  --label 100:/path/vat.nii.gz \
  --label 101:/path/sat.nii.gz

2) Referans geometriyi zorlamak için:
python scripts/merge_nifti_labels.py \
  --reference /path/any_volume.nii.gz \
  --out merged_labels.nii.gz \
  --label 55:/path/psoas_left.nii.gz ...

Not: Çakışan voxellerde son gelen etiket kazanır (priority=order).
"""


def parse_label_pair(s: str) -> Tuple[int, Path]:
    # format: ID:/path/to/file.nii.gz
    try:
        k, v = s.split(":", 1)
        return int(k), Path(v)
    except Exception:
        raise argparse.ArgumentTypeError(f"Geçersiz --label değeri: {s} (beklenen: id:/path)")


def main():
    ap = argparse.ArgumentParser(description="Merge per-class NIfTI to multi-label NIfTI")
    ap.add_argument("--label", action="append", type=parse_label_pair, help="id:/path/to/nii", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default="", help="Geometri referansı; boşsa ilk girdi temel alınır")
    args = ap.parse_args()

    label_pairs: List[Tuple[int, Path]] = args.label
    if not label_pairs:
        print("[MERGE] HATA: --label girilmedi")
        return 2

    ref_img = None
    if args.reference:
        ref_img = sitk.ReadImage(str(Path(args.reference)))
        ref_shape = sitk.GetArrayFromImage(ref_img).shape
    else:
        # İlk girdi referans
        ref_img = sitk.ReadImage(str(label_pairs[0][1]))
        ref_shape = sitk.GetArrayFromImage(ref_img).shape

    out = np.zeros(ref_shape, dtype=np.int16)

    for lab_id, p in label_pairs:
        if not p.exists():
            print("[MERGE] WARN: yok sayılıyor (bulunamadı)", p)
            continue
        img = sitk.ReadImage(str(p))
        arr = sitk.GetArrayFromImage(img)
        if arr.shape != ref_shape:
            print("[MERGE] HATA: geometri uyumsuz", p, arr.shape, "!=", ref_shape)
            return 3
        mask = arr > 0
        out[mask] = np.int16(lab_id)

    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(ref_img)
    sitk.WriteImage(out_img, str(Path(args.out)))
    print("[MERGE] yazıldı:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
