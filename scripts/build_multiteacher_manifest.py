import argparse
import csv
from pathlib import Path

"""
Girdi: images_root (PNG dilimler), opsiyonel gt_root/ts_root/c2c_root (mask PNG'leri)
Çıktı: manifest CSV (image,gt_mask,ts_mask,c2c_mask,w_gt,w_ts,w_c2c)

Not: Dosya adları eşleme kuralı: <name>.png
"""


def find_mask(mask_root: Path, base: str) -> str:
    if not mask_root:
        return ""
    p = mask_root / f"{base}.png"
    return str(p) if p.exists() else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-root", default="")
    ap.add_argument("--ts-root", default="")
    ap.add_argument("--c2c-root", default="")
    ap.add_argument("--w-gt", type=float, default=1.0)
    ap.add_argument("--w-ts", type=float, default=0.3)
    ap.add_argument("--w-c2c", type=float, default=0.3)
    args = ap.parse_args()

    images_root = Path(args.images_root)
    gt_root = Path(args.gt_root) if args.gt_root else None
    ts_root = Path(args.ts_root) if args.ts_root else None
    c2c_root = Path(args.c2c_root) if args.c2c_root else None

    images = sorted(images_root.glob("**/*.png"))
    if not images:
        print("[MANIFEST] PNG bulunamadı:", images_root)
        return 1

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=[
                "image", "gt_mask", "ts_mask", "c2c_mask", "w_gt", "w_ts", "w_c2c"
            ]
        )
        w.writeheader()
        for img in images:
            base = img.stem
            row = {
                "image": str(img),
                "gt_mask": find_mask(gt_root, base) if gt_root else "",
                "ts_mask": find_mask(ts_root, base) if ts_root else "",
                "c2c_mask": find_mask(c2c_root, base) if c2c_root else "",
                "w_gt": args.w_gt,
                "w_ts": args.w_ts,
                "w_c2c": args.w_c2c,
            }
            w.writerow(row)
    print("[MANIFEST] yazıldı:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
