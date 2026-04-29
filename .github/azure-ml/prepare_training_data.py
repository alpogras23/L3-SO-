#!/usr/bin/env python3
"""
Azure ML: Prepare Training Data for DL Model
- AMOS22 NIfTI ve TotalSegmentator maskelerinden L3 slices çıkart
- 3-channel input (HU + TS + vertebra) ve 3-channel output (VFA + PMA + inner_abdomen) hazırla
- PyTorch dataset'e uygun şekilde organize et
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple
import traceback

import numpy as np
import cv2
import SimpleITK as sitk
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core_mini import (
        load_hu,
        find_l3_slice_index,
        detect_vertebra_center,
        compute_inner_abdomen_mask,
        compute_vfa,
        estimate_psoas_mask,
        compute_pma,
        get_pixel_area_mm2,
    )
except ImportError:
    from l3_vfa_pma_core_v3_7 import (
        load_hu,
        find_l3_slice_index,
        detect_vertebra_center,
        compute_inner_abdomen_mask,
        compute_vfa,
        estimate_psoas_mask,
        compute_pma,
        get_pixel_area_mm2,
    )


class TrainingDataPreparer:
    """AMOS22 verilerinden training dataset hazırla"""

    def __init__(self, output_dir: Path, target_size: Tuple[int, int] = (512, 512)):
        self.output_dir = output_dir
        self.target_size = target_size
        self.train_data = []
        self.val_data = []
        self.metadata = []

    def prepare_case(
        self, amos_nii_path: Path, ts_mask_path: Path = None
    ) -> Tuple[bool, dict]:
        """
        Bir AMOS22 vakasını training data olarak hazırla

        Returns:
            (success: bool, metadata: dict)
        """
        case_id = amos_nii_path.stem.replace(".nii", "")
        meta = {"case_id": case_id, "status": "processing"}

        try:
            # Load HU volume
            hu_vol, spacing, affine = load_hu(str(amos_nii_path))
            if hu_vol is None:
                meta["status"] = "failed_hu_load"
                return False, meta

            # Find L3
            z_l3, vb_conf = find_l3_slice_index(hu_vol)
            if vb_conf < 0.5:
                meta["status"] = "low_vb_confidence"
                meta["vb_conf"] = float(vb_conf)
                return False, meta

            # Extract L3 slice
            hu_l3 = hu_vol[z_l3, :, :]

            # Load TS mask
            ts_seg = None
            if ts_mask_path and ts_mask_path.exists():
                try:
                    ts_img = sitk.ReadImage(str(ts_mask_path))
                    ts_vol = sitk.GetArrayFromImage(ts_img)
                    if ts_vol.ndim == 3:
                        ts_seg = ts_vol[z_l3, :, :]
                    else:
                        ts_seg = ts_vol
                except Exception as e:
                    print(f"  ⚠️ TS mask yüklenemedi: {e}")

            # Detect vertebra center
            vb_center, vb_mask = detect_vertebra_center(hu_l3)
            if vb_center is None:
                meta["status"] = "vertebra_detection_failed"
                return False, meta

            # Compute segmentations
            inner_mask = compute_inner_abdomen_mask(hu_l3, ts_seg, vb_center)
            psoas_mask = estimate_psoas_mask(hu_l3, vb_center)

            # Compute masks
            vfa_result = compute_vfa(hu_l3, inner_mask, get_pixel_area_mm2(spacing[:2]))
            pma_result = compute_pma(hu_l3, psoas_mask, get_pixel_area_mm2(spacing[:2]))

            vfa_mask = vfa_result["vfa_mask"]
            pma_mask = pma_result["pma_mask"]

            # Resize to 512x512
            hu_resized = cv2.resize(
                hu_l3.astype(np.float32), self.target_size, interpolation=cv2.INTER_LINEAR
            )
            ts_resized = (
                cv2.resize(ts_seg.astype(np.float32), self.target_size, interpolation=cv2.INTER_NEAREST)
                if ts_seg is not None
                else np.zeros(self.target_size, dtype=np.float32)
            )
            vb_resized = cv2.resize(
                vb_mask.astype(np.float32), self.target_size, interpolation=cv2.INTER_NEAREST
            )
            vfa_resized = cv2.resize(
                vfa_mask.astype(np.float32), self.target_size, interpolation=cv2.INTER_NEAREST
            )
            pma_resized = cv2.resize(
                pma_mask.astype(np.float32), self.target_size, interpolation=cv2.INTER_NEAREST
            )
            inner_resized = cv2.resize(
                inner_mask.astype(np.float32), self.target_size, interpolation=cv2.INTER_NEAREST
            )

            # Normalize HU
            hu_normalized = np.clip(hu_resized, -150, 250)
            hu_normalized = (hu_normalized + 150) / 400.0

            # Stack input channels: (3, 512, 512)
            input_3ch = np.stack([hu_normalized, (ts_resized > 0).astype(np.float32), (vb_resized > 0).astype(np.float32)], axis=0)

            # Stack output channels: (3, 512, 512)
            output_3ch = np.stack(
                [(vfa_resized > 0).astype(np.float32), (pma_resized > 0).astype(np.float32), (inner_resized > 0).astype(np.float32)],
                axis=0,
            )

            # Save as NPZ
            case_dir = self.output_dir / case_id
            case_dir.mkdir(exist_ok=True, parents=True)

            input_path = case_dir / "input.npy"
            output_path = case_dir / "output.npy"

            np.save(input_path, input_3ch.astype(np.float32))
            np.save(output_path, output_3ch.astype(np.uint8))

            meta["status"] = "success"
            meta["l3_index"] = int(z_l3)
            meta["vb_confidence"] = float(vb_conf)
            meta["input_shape"] = list(input_3ch.shape)
            meta["output_shape"] = list(output_3ch.shape)

            return True, meta

        except Exception as e:
            meta["status"] = "failed"
            meta["error"] = str(e)
            meta["traceback"] = traceback.format_exc()
            return False, meta

    def finalize(self, train_ratio: float = 0.8):
        """Finalize dataset - train/val split"""
        # Metadata dosyasını kaydet
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        # Train/val split
        successful_cases = [m for m in self.metadata if m["status"] == "success"]
        n_train = int(len(successful_cases) * train_ratio)

        train_list = [m["case_id"] for m in successful_cases[:n_train]]
        val_list = [m["case_id"] for m in successful_cases[n_train:]]

        split_info = {
            "total": len(successful_cases),
            "train": len(train_list),
            "val": len(val_list),
            "train_ids": train_list,
            "val_ids": val_list,
        }

        split_path = self.output_dir / "split.json"
        with open(split_path, "w") as f:
            json.dump(split_info, f, indent=2)

        return split_info


def find_amos_cases(data_dir: Path) -> list:
    """AMOS22 NIfTI dosyalarını bul"""
    nii_files = list(data_dir.glob("*.nii.gz")) + list(data_dir.glob("*.nii"))
    return sorted(nii_files)


def find_ts_mask(case_id: str, ts_root: Path) -> Path:
    """TotalSegmentator maskesini bul"""
    candidates = [
        ts_root / case_id / "abdominal_muscles.nii.gz",
        ts_root / case_id / "abdominal_muscles.nii",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main(args):
    print("=" * 60)
    print("🚀 Azure ML Training Data Preparation")
    print("=" * 60)

    amos_root = Path(args.amos_root)
    ts_root = Path(args.ts_root)
    output_dir = Path(args.output_dir)
    max_cases = args.max_cases

    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"📂 AMOS22: {amos_root}")
    print(f"📂 TS masks: {ts_root}")
    print(f"📂 Output: {output_dir}")

    # Find cases
    amos_cases = find_amos_cases(amos_root)
    print(f"📦 {len(amos_cases)} AMOS22 cases found")

    if max_cases:
        amos_cases = amos_cases[:max_cases]
        print(f"📌 Processing first {max_cases} cases")

    # Prepare
    preparer = TrainingDataPreparer(output_dir)

    successful = 0
    for amos_path in tqdm(amos_cases, desc="Preparing training data"):
        case_id = amos_path.stem.replace(".nii", "")
        ts_path = find_ts_mask(case_id, ts_root)

        success, meta = preparer.prepare_case(amos_path, ts_path)
        preparer.metadata.append(meta)

        if success:
            successful += 1

    # Finalize
    split_info = preparer.finalize(train_ratio=0.8)

    print("\n" + "=" * 60)
    print("✅ Training data preparation complete!")
    print(f"  Total successful: {successful}/{len(amos_cases)}")
    print(f"  Train/Val split: {split_info['train']}/{split_info['val']}")
    print(f"📊 Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data for DL model")
    parser.add_argument("--amos_root", type=str, required=True, help="AMOS22 directory")
    parser.add_argument("--ts_root", type=str, required=True, help="TS masks directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--max_cases", type=int, default=50, help="Max cases to process")

    args = parser.parse_args()
    main(args)
