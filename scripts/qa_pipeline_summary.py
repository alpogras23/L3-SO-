import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def read_png(p: Path) -> np.ndarray:
    img = sitk.ReadImage(str(p))
    return sitk.GetArrayFromImage(img)


def main():
    ap = argparse.ArgumentParser(description="Export edilmiş PNG ve maske klasörlerinden QA özeti üretir")
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--gt-root", default="")
    ap.add_argument("--ts-root", default="")
    ap.add_argument("--c2c-root", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--print-head", type=int, default=20)
    args = ap.parse_args()

    images_root = Path(args.images_root)
    gt_root = Path(args.gt_root) if args.gt_root else None
    ts_root = Path(args.ts_root) if args.ts_root else None
    c2c_root = Path(args.c2c_root) if args.c2c_root else None

    rows = []
    for img_p in sorted(images_root.glob("*.png")):
        name = img_p.stem
        try:
            arr = read_png(img_p)
            if arr.ndim == 3:  # sitk png -> (1,H,W)
                arr = arr[0]
            h, w = arr.shape
        except Exception:
            h = w = 0
        row = {
            "name": name,
            "image_path": str(img_p),
            "H": h,
            "W": w,
        }
        # C2C
        if c2c_root:
            c2c_p = c2c_root / f"{name}.png"
            if c2c_p.exists():
                try:
                    m = read_png(c2c_p)
                    if m.ndim == 3:
                        m = m[0]
                    row["c2c_mask"] = str(c2c_p)
                    row["c2c_nonzero"] = int((m > 0).sum())
                except Exception:
                    row["c2c_mask"] = "read_error"
                    row["c2c_nonzero"] = -1
            else:
                row["c2c_mask"] = ""
                row["c2c_nonzero"] = 0
        # GT
        if gt_root:
            gt_p = gt_root / f"{name}.png"
            if gt_p.exists():
                try:
                    m = read_png(gt_p)
                    if m.ndim == 3:
                        m = m[0]
                    row["gt_mask"] = str(gt_p)
                    row["gt_nonzero"] = int((m > 0).sum())
                except Exception:
                    row["gt_mask"] = "read_error"
                    row["gt_nonzero"] = -1
            else:
                row["gt_mask"] = ""
                row["gt_nonzero"] = 0
        # TS
        if ts_root:
            ts_p = ts_root / f"{name}.png"
            if ts_p.exists():
                try:
                    m = read_png(ts_p)
                    if m.ndim == 3:
                        m = m[0]
                    row["ts_mask"] = str(ts_p)
                    row["ts_nonzero"] = int((m > 0).sum())
                except Exception:
                    row["ts_mask"] = "read_error"
                    row["ts_nonzero"] = -1
            else:
                row["ts_mask"] = ""
                row["ts_nonzero"] = 0
        rows.append(row)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name", "image_path", "H", "W",
        "gt_mask", "gt_nonzero",
        "ts_mask", "ts_nonzero",
        "c2c_mask", "c2c_nonzero",
    ]
    # Filter by provided roots
    if not gt_root:
        fieldnames.remove("gt_mask")
        fieldnames.remove("gt_nonzero")
    if not ts_root:
        fieldnames.remove("ts_mask")
        fieldnames.remove("ts_nonzero")
    if not c2c_root:
        fieldnames.remove("c2c_mask")
        fieldnames.remove("c2c_nonzero")

    with open(out_p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print("[QA] yazıldı:", out_p)
    for r in rows[: max(0, int(args.print_head))]:
        print("[QA]", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
