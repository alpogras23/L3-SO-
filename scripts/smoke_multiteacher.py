from pathlib import Path

import cv2
import numpy as np

"""
Küçük sentetik bir veri kümesi üretir ve manifest yazar.
"""

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "psoas_ml" / "data" / "images"
GT = ROOT / "psoas_ml" / "data" / "gt_masks"
TS = ROOT / "psoas_ml" / "data" / "ts_masks"
C2C = ROOT / "psoas_ml" / "data" / "c2c_masks"
MAN = ROOT / "psoas_ml" / "data" / "multiteacher_manifest.csv"


def draw_circle(h=128, w=128, r=20, off=0):
    img = np.zeros((h, w), np.uint8)
    cv2.circle(img, (w // 2 + off, h // 2), r, 255, -1)
    return img


def main():
    for d in [IMG, GT, TS, C2C]:
        d.mkdir(parents=True, exist_ok=True)

    rows = [
        ("a", 0, 0),
        ("b", 5, -5),
        ("c", -5, 5),
        ("d", 3, 3),
    ]
    with open(MAN, "w", encoding="utf-8") as fh:
        fh.write("image,gt_mask,ts_mask,c2c_mask,w_gt,w_ts,w_c2c\n")
        for name, off_ts, off_c2c in rows:
            img = draw_circle(128, 128, 24, 0)
            gt = draw_circle(128, 128, 24, 0)
            ts = draw_circle(128, 128, 22, off_ts)
            c2c = draw_circle(128, 128, 20, off_c2c)
            cv2.imwrite(str(IMG / f"{name}.png"), img)
            cv2.imwrite(str(GT / f"{name}.png"), gt)
            cv2.imwrite(str(TS / f"{name}.png"), ts)
            cv2.imwrite(str(C2C / f"{name}.png"), c2c)
            fh.write(
                f"{IMG / f'{name}.png'},{GT / f'{name}.png'},{TS / f'{name}.png'},{C2C / f'{name}.png'},1.0,0.3,0.3\n"
            )
    print("[SMOKE] Manifest hazır:", MAN)
    print("Örnek eğitim komutu:")
    print(
        "python psoas_ml/multiteacher_train.py --manifest",
        MAN,
        "--epochs 1 --batch-size 2 --threads 2 --out psoas_ml/ckpts",
    )


if __name__ == "__main__":
    main()
