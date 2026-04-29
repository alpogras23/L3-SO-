import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def read_png_gray(p: Path) -> np.ndarray:
    img = sitk.ReadImage(str(p))
    arr = sitk.GetArrayFromImage(img)
    # Expect shape (1,H,W) or (H,W)
    if arr.ndim == 3:
        arr = arr[0]
    return arr.astype(np.float32), img


def write_png_rgb(p: Path, rgb: np.ndarray, ref_img: sitk.Image):
    # rgb: (H,W,3) uint8
    # Build a vector image with same spacing/origin/direction
    img = sitk.GetImageFromArray(rgb, isVector=True)
    img.SetSpacing(ref_img.GetSpacing())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    sitk.WriteImage(img, str(p))


def overlay_colors(base: np.ndarray, masks: dict, alpha: float = 0.6) -> np.ndarray:
    """Return an RGB overlay image.
    base: (H,W) float32 0..255
    masks: name -> (mask ndarray bool, color tuple RGB 0..255)
    """
    H, W = base.shape
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    for name, (m, color) in masks.items():
        if m is None:
            continue
        if m.ndim == 3:
            m = m[0]
        if m.shape != (H, W):
            continue
        m_bool = m > 0
        if not np.any(m_bool):
            continue
        col = np.array(color, dtype=np.float32)[None, None, :]
        # alpha blend only where mask
        mask3 = np.repeat(m_bool[:, :, None], 3, axis=2)
        rgb[mask3] = (1 - alpha) * rgb[mask3] + alpha * col.repeat(np.count_nonzero(mask3) // 3, axis=0).reshape(-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def main():
    ap = argparse.ArgumentParser(description="Görüntü+maskelerden renkli overlay PNG üretir")
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--gt-root", default="")
    ap.add_argument("--ts-root", default="")
    ap.add_argument("--c2c-root", default="")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--alpha", type=float, default=0.6)
    args = ap.parse_args()

    images_root = Path(args.images_root)
    gt_root = Path(args.gt_root) if args.gt_root else None
    ts_root = Path(args.ts_root) if args.ts_root else None
    c2c_root = Path(args.c2c_root) if args.c2c_root else None
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_p in sorted(images_root.glob("*.png")):
        base, ref = read_png_gray(img_p)
        name = img_p.stem

        def read_mask(root: Path | None, nm: str):
            if not root:
                return None
            p = root / f"{nm}.png"
            if not p.exists():
                return None
            try:
                arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
                if arr.ndim == 3:
                    arr = arr[0]
                return arr
            except Exception:
                return None

        masks = {}
        # Renkler: C2C=red, GT=green, TS=blue
        c2c = read_mask(c2c_root, name)
        gt = read_mask(gt_root, name)
        ts = read_mask(ts_root, name)
        masks["c2c"] = (c2c, (255, 0, 0))
        masks["gt"] = (gt, (0, 255, 0))
        masks["ts"] = (ts, (0, 0, 255))

        rgb = overlay_colors(base, masks, alpha=args.alpha)
        out_p = out_root / f"{name}_overlay.png"
        write_png_rgb(out_p, rgb, ref)
        print("[OVERLAY]", out_p)
        count += 1

    print(f"[OVERLAY] tamam: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
