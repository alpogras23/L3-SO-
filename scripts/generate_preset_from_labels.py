import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk

"""
Label NIfTI dosyalarından ID histogramı üretir ve istenirse bir preset JSON'u günceller/oluşturur.

Örnek:
python scripts/generate_preset_from_labels.py \
  --nii data/teachers/c2c/CASE_001/labels.nii.gz \
  --nii data/teachers/ts/CASE_001/labels.nii.gz \
  --out-json scripts/label_presets/tbcc.json \
  --set-c2c-l3-id 55 \
  --set-c2c-psoas-ids 55,56 \
  --set-c2c-vat-id 100 --set-c2c-sat-id 101 \
  --set-gt-psoas-ids 55,56 --set-ts-psoas-ids 55,56
"""


def parse_ids_csv(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Inspect labels and optionally write preset JSON")
    ap.add_argument("--nii", action="append", default=[], help="Label NIfTI path (repeatable)")
    ap.add_argument("--out-json", default="", help="Preset JSON to write/update")
    # Setters for preset fields
    ap.add_argument("--set-c2c-l3-id", type=int, default=None)
    ap.add_argument("--set-c2c-vat-id", type=int, default=None)
    ap.add_argument("--set-c2c-sat-id", type=int, default=None)
    ap.add_argument("--set-c2c-psoas-ids", default="")
    ap.add_argument("--set-gt-psoas-ids", default="")
    ap.add_argument("--set-ts-psoas-ids", default="")
    ap.add_argument("--set-w-gt", type=float, default=None)
    ap.add_argument("--set-w-ts", type=float, default=None)
    ap.add_argument("--set-w-c2c", type=float, default=None)
    args = ap.parse_args()

    # Aggregate histograms
    counts: Dict[int, int] = {}
    for p in args.nii:
        path = Path(p)
        if not path.exists():
            print("[PRESET] WARN: yok sayılıyor (bulunamadı)", path)
            continue
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        vals, cnts = np.unique(arr, return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            counts[int(v)] = counts.get(int(v), 0) + int(c)

    if counts:
        print("[PRESET] Birleşik label histogramı (count, id):")
        for c, v in sorted(((c, v) for v, c in counts.items()), reverse=True)[:40]:
            print(f"{c:12d}  {v}")
    else:
        print("[PRESET] Uyarı: Histogram boş (geçerli NIfTI bulunamadı)")

    if not args.out_json:
        return 0

    preset_path = Path(args.out_json)
    if preset_path.exists():
        try:
            cfg = json.loads(preset_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    else:
        cfg = {}

    def ensure(obj: Dict, key: str, default):
        if key not in obj or not isinstance(obj[key], type(default)):
            obj[key] = default

    ensure(cfg, "c2c", {})
    ensure(cfg, "gt", {})
    ensure(cfg, "ts", {})
    ensure(cfg, "weights", {})

    # Apply setters
    if args.set_c2c_l3_id is not None:
        cfg["c2c"]["l3_id"] = int(args.set_c2c_l3_id)
    if args.set_c2c_vat_id is not None:
        cfg["c2c"]["vat_id"] = int(args.set_c2c_vat_id)
    if args.set_c2c_sat_id is not None:
        cfg["c2c"]["sat_id"] = int(args.set_c2c_sat_id)
    c2c_p = parse_ids_csv(args.set_c2c_psoas_ids)
    if c2c_p:
        cfg["c2c"]["psoas_ids"] = c2c_p
    gt_p = parse_ids_csv(args.set_gt_psoas_ids)
    if gt_p:
        cfg["gt"]["psoas_ids"] = gt_p
    ts_p = parse_ids_csv(args.set_ts_psoas_ids)
    if ts_p:
        cfg["ts"]["psoas_ids"] = ts_p
    if args.set_w_gt is not None:
        cfg["weights"]["w_gt"] = float(args.set_w_gt)
    if args.set_w_ts is not None:
        cfg["weights"]["w_ts"] = float(args.set_w_ts)
    if args.set_w_c2c is not None:
        cfg["weights"]["w_c2c"] = float(args.set_w_c2c)

    preset_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[PRESET] Yazıldı:", preset_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
