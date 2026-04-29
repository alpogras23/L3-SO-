import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import SimpleITK as sitk


def read_dicom_series(series_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {series_dir}")
    reader.SetFileNames(files)
    img = reader.Execute()
    return img


def load_label_map(nii_path: Path) -> sitk.Image:
    return sitk.ReadImage(str(nii_path))


def to_numpy(img: sitk.Image) -> np.ndarray:
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    return arr


def window_hu_to_u8(arr_hu: np.ndarray, wlow=-150, whigh=250) -> np.ndarray:
    arr = np.clip(arr_hu, wlow, whigh)
    arr = (arr - wlow) / max(1e-3, (whigh - wlow))
    arr = (arr * 255.0).round().astype(np.uint8)
    return arr


def find_l3_index_from_label(label_vol: np.ndarray, l3_label_id: int) -> int:
    # En büyük alanlı dilimi seç
    areas = (label_vol == l3_label_id).sum(axis=(1, 2))
    idx = int(np.argmax(areas))
    return idx


def parse_multi_int(s: Optional[str]) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Export L3 image + C2C masks as 2D PNGs")
    ap.add_argument("--dicom-series", required=True, help="DICOM series folder")
    ap.add_argument("--out-images", required=True)
    ap.add_argument("--out-c2c", required=True)
    ap.add_argument("--out-ts", default="", help="output dir for TS masks (optional)")
    ap.add_argument("--out-gt", default="", help="output dir for GT masks (optional)")
    ap.add_argument("--prefix", default="case")
    ap.add_argument("--l3-index", type=int, default=-1, help="Override L3 index (z)")
    # C2C label map and IDs
    ap.add_argument("--c2c-label-map", default="", help="C2C multi-label NIfTI path")
    ap.add_argument("--label-c2c-l3", type=int, default=-1, help="Label id for L3 vertebra slice detection")
    ap.add_argument("--label-c2c-vat", type=int, default=-1)
    ap.add_argument("--label-c2c-sat", type=int, default=-1)
    ap.add_argument("--label-c2c-psoas", default="", help="comma-separated ids, e.g. '55,56'")
    ap.add_argument("--c2c-base-from-psoas", action="store_true", help="also write {prefix}.png mask equal to psoas (manifest uyumu için)")
    # Optional GT label map
    ap.add_argument("--gt-label-map", default="", help="GT multi-label NIfTI path (optional)")
    ap.add_argument("--label-gt-psoas", default="", help="comma-separated ids for GT psoas")
    # Optional TS label map
    ap.add_argument("--ts-label-map", default="", help="TS multi-label NIfTI path (optional)")
    ap.add_argument("--label-ts-psoas", default="", help="comma-separated ids for TS psoas")
    args = ap.parse_args()

    series_dir = Path(args.dicom_series)
    out_images = Path(args.out_images)
    out_c2c = Path(args.out_c2c)
    out_images.mkdir(parents=True, exist_ok=True)
    out_c2c.mkdir(parents=True, exist_ok=True)
    out_gt = Path(args.out_gt) if args.out_gt else None
    if out_gt:
        out_gt.mkdir(parents=True, exist_ok=True)
    out_ts = Path(args.out_ts) if args.out_ts else None
    if out_ts:
        out_ts.mkdir(parents=True, exist_ok=True)

    vol = read_dicom_series(series_dir)
    arr = to_numpy(vol).astype(np.float32)  # z,y,x, may already include rescale

    # Try to derive HU if rescale present: SimpleITK applies rescale; assume arr is in intensities
    # Window to display
    arr_u8 = window_hu_to_u8(arr, -150, 250)

    z_idx = args.l3_index
    c2c_map = None
    if args.c2c_label_map:
        c2c_map = load_label_map(Path(args.c2c_label_map))
        c2c_np = to_numpy(c2c_map)
        if z_idx < 0 and args.label_c2c_l3 >= 0:
            z_idx = find_l3_index_from_label(c2c_np, args.label_c2c_l3)

    if z_idx < 0:
        # Fallback: orta dilim
        z_idx = int(arr.shape[0] // 2)

    # Write image PNG
    img_png = out_images / f"{args.prefix}.png"
    cv2.imwrite(str(img_png), arr_u8[z_idx])

    # Write C2C masks
    if c2c_map is not None:
        c2c_np = to_numpy(c2c_map)
        h, w = c2c_np.shape[1:]
        # psoas could be multiple ids
        psoas_ids = parse_multi_int(args.label_c2c_psoas)
        if psoas_ids:
            m = np.isin(c2c_np[z_idx], np.array(psoas_ids, dtype=c2c_np.dtype))
            psoas_png = out_c2c / f"{args.prefix}_psoas.png"
            cv2.imwrite(str(psoas_png), (m.astype(np.uint8) * 255))
            if args.c2c_base_from_psoas:
                base_png = out_c2c / f"{args.prefix}.png"
                cv2.imwrite(str(base_png), (m.astype(np.uint8) * 255))
        if args.label_c2c_vat >= 0:
            m = (c2c_np[z_idx] == args.label_c2c_vat)
            cv2.imwrite(str(out_c2c / f"{args.prefix}_vat.png"), (m.astype(np.uint8) * 255))
        if args.label_c2c_sat >= 0:
            m = (c2c_np[z_idx] == args.label_c2c_sat)
            cv2.imwrite(str(out_c2c / f"{args.prefix}_sat.png"), (m.astype(np.uint8) * 255))

    # Write GT masks (optional)
    if args.gt_label_map and out_gt is not None:
        gt_np = to_numpy(load_label_map(Path(args.gt_label_map)))
        gt_psoas_ids = parse_multi_int(args.label_gt_psoas)
        if gt_psoas_ids:
            m = np.isin(gt_np[z_idx], np.array(gt_psoas_ids, dtype=gt_np.dtype))
            cv2.imwrite(str(out_gt / f"{args.prefix}.png"), (m.astype(np.uint8) * 255))

    # Write TS masks (optional)
    if args.ts_label_map and out_ts is not None:
        ts_np = to_numpy(load_label_map(Path(args.ts_label_map)))
        ts_psoas_ids = parse_multi_int(args.label_ts_psoas)
        if ts_psoas_ids:
            m = np.isin(ts_np[z_idx], np.array(ts_psoas_ids, dtype=ts_np.dtype))
            cv2.imwrite(str(out_ts / f"{args.prefix}.png"), (m.astype(np.uint8) * 255))

    print("[EXPORT] Image:", img_png)
    print("[EXPORT] Masks dir:", out_c2c)


if __name__ == "__main__":
    raise SystemExit(main())
