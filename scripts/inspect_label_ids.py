import argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk

"""
NIfTI label haritasındaki benzersiz ID'leri ve voxel sayılarını raporlar.
Kullanım:
python scripts/inspect_label_ids.py --nii path/to/labels.nii.gz --top 20
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nii", required=True)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    img = sitk.ReadImage(str(Path(args.nii)))
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    vals, cnts = np.unique(arr, return_counts=True)
    pairs = sorted(zip(cnts, vals), reverse=True)
    print("Top labels (count, id):")
    for c, v in pairs[: args.top]:
        print(f"{c:10d}  {int(v)}")


if __name__ == "__main__":
    main()
