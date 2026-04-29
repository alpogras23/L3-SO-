#!/usr/bin/env python3
"""
TWO-STAGE L3 PSOAS SEGMENTATION
================================

STAGE 1: Find L3 slice (task="total", fast=True, LOW RAM)
STAGE 2: Segment psoas on L3 slice only (task="abdominal_muscles", fast=False, LOW RAM)

This solves the RAM exhaustion problem by processing only a single slice.
"""

import sys
import json
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import nibabel as nib
import cv2
from scipy.signal import find_peaks


def find_l3_slice_by_hu_threshold(ct_path: str) -> Optional[int]:
    """
    Find L3 slice using HU thresholding for vertebrae (bone density).
    Vertebrae have HU values around 200-1000.
    
    Args:
        ct_path: Path to CT volume
        
    Returns:
        L3 slice index or None
    """
    try:
        ct_nib = nib.load(ct_path)
        ct_data = ct_nib.get_fdata()
        
        # Find slices with high bone density (vertebrae)
        bone_threshold = 200  # HU threshold for bone
        bone_slices = []
        
        for z in range(ct_data.shape[2]):
            slice_data = ct_data[:, :, z]
            bone_pixels = (slice_data > bone_threshold).sum()
            bone_slices.append(bone_pixels)
        
        # Find peaks in bone density (vertebrae locations)
        bone_array = np.array(bone_slices)
        peaks, _ = find_peaks(bone_array, height=np.mean(bone_array), distance=5)
        
        if len(peaks) >= 3:  # At least L3, L4, L5
            # L3 is typically the 3rd lumbar vertebra from bottom
            # Sort peaks by slice position (lower slices = more inferior)
            sorted_peaks = sorted(peaks)
            if len(sorted_peaks) >= 3:
                l3_idx = sorted_peaks[-3]  # 3rd from bottom
                print(f"  ✅ L3 detected by HU threshold: slice {l3_idx} (bone peaks: {len(peaks)})")
                return int(l3_idx)
        
        print(f"  ⚠️  HU-based L3 detection failed: found {len(peaks)} bone peaks")
        return None
        
    except Exception as e:
        print(f"  ❌ HU-based L3 detection failed: {e}")
        return None


def find_psoas_by_hu_and_position(hu_slice: np.ndarray, vertebra_center_x: int, slice_height: int, slice_width: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find psoas muscles using HU thresholding and anatomical position relative to vertebra.
    
    Args:
        hu_slice: HU values for the slice
        vertebra_center_x: X coordinate of vertebra center
        slice_height: Height of slice
        slice_width: Width of slice
        
    Returns:
        Tuple of (left_psoas_mask, right_psoas_mask)
    """
    # Psoas HU range: muscle tissue
    muscle_mask = (hu_slice >= -29) & (hu_slice <= 150)
    
    # Create left psoas mask
    left_psoas = np.zeros_like(muscle_mask, dtype=np.uint8)
    if vertebra_center_x > 50:  # Ensure we have space on the left
        left_start = max(0, vertebra_center_x - 150)
        left_end = max(0, vertebra_center_x - 50)
        left_psoas[:, left_start:left_end] = muscle_mask[:, left_start:left_end]
    
    # Create right psoas mask  
    right_psoas = np.zeros_like(muscle_mask, dtype=np.uint8)
    if vertebra_center_x < slice_width - 50:  # Ensure we have space on the right
        right_start = min(slice_width, vertebra_center_x + 50)
        right_end = min(slice_width, vertebra_center_x + 150)
        right_psoas[:, right_start:right_end] = muscle_mask[:, right_start:right_end]
    
    # Clean up masks (remove small components, morphological operations)
    kernel = np.ones((3, 3), np.uint8)
    left_psoas = cv2.morphologyEx(left_psoas.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    right_psoas = cv2.morphologyEx(right_psoas.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    # Keep only largest component for each psoas
    left_psoas = keep_largest_component(left_psoas)
    right_psoas = keep_largest_component(right_psoas)
    
    return left_psoas, right_psoas


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask."""
    if mask.sum() == 0:
        return mask
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    
    if num_labels <= 1:
        return mask
    
    # Find largest component (excluding background)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = (labels == largest_label).astype(np.uint8)
    
    return largest_mask


def extract_single_slice(ct_path: str, slice_idx: int, output_path: str) -> bool:
    """
    Extract a single slice from CT volume and save as 3D volume (1 slice).
    
    Args:
        ct_path: Path to full CT NII.GZ
        slice_idx: Z-axis slice index
        output_path: Path to save single-slice NII.GZ
        
    Returns:
        Success status
    """
    try:
        ct_nib = nib.load(ct_path)
        ct_data = ct_nib.get_fdata()
        
        # Extract single slice but keep 3D shape (X, Y, 1)
        single_slice = ct_data[:, :, slice_idx:slice_idx+1]
        
        # Create new NIfTI with same affine
        slice_nib = nib.Nifti1Image(single_slice, ct_nib.affine, ct_nib.header)
        nib.save(slice_nib, output_path)
        
        print(f"  ✅ Extracted slice {slice_idx}: {single_slice.shape}")
        return True
        
    except Exception as e:
        print(f"  ❌ Slice extraction failed: {e}")
        return False


def load_mask_slice(mask_path: Path, slice_idx: int) -> Optional[np.ndarray]:
    """Load a 3D mask and return the requested slice (axis=0)."""
    try:
        mask_nib = nib.load(str(mask_path))
        mask_data = mask_nib.get_fdata()
        if mask_data.ndim == 4:
            mask_data = mask_data[..., 0]
        if slice_idx >= mask_data.shape[0]:
            return None
        return (mask_data[slice_idx, :, :] > 0).astype(np.uint8)
    except Exception as e:
        print(f"  ⚠️  Failed to load mask slice from {mask_path}: {e}")
        return None


def build_fat_mask(ts_dir: Path, slice_idx: int, hu_slice: np.ndarray) -> np.ndarray:
    """
    Try to get fat mask from TotalSegmentator output; fallback to HU thresholding.
    Fat HU range: [-190, -30].
    """
    candidates = [
        "torso_fat.nii.gz",
        "visceral_fat.nii.gz",
        "adipose_intra_abdominal.nii.gz",
        "vat.nii.gz",
    ]
    for name in candidates:
        for cand in ts_dir.rglob(name):
            mask_slice = load_mask_slice(cand, slice_idx)
            if mask_slice is not None and mask_slice.sum() > 0:
                print(f"  ✅ FAT mask from TS: {cand.name} ({mask_slice.sum()} px)")
                return mask_slice
    # Fallback: HU threshold
    fat_mask = ((hu_slice >= -190) & (hu_slice <= -30)).astype(np.uint8)
    print(f"  ⚠️  FAT mask via HU threshold ({fat_mask.sum()} px)")
    return fat_mask


def build_muscle_mask(hu_slice: np.ndarray) -> np.ndarray:
    """
    Simple HU-based muscle mask.
    Muscle HU range: [-29, 150]; also require body mask (HU>-300).
    """
    body = (hu_slice > -300)
    muscle = (hu_slice >= -29) & (hu_slice <= 150)
    mask = (body & muscle).astype(np.uint8)
    print(f"  ✅ Muscle mask via HU range ({mask.sum()} px)")
    return mask


def stage1_find_l3(ct_path: str, temp_dir: str) -> Optional[int]:
    """
    STAGE 1: Find L3 slice using HU-based bone detection (no TotalSegmentator dependency).
    
    Args:
        ct_path: Path to full CT volume
        temp_dir: Temporary directory (not used for HU method)
        
    Returns:
        L3 slice index or None
    """
    print("  🔍 STAGE 1: Finding L3 slice using HU-based bone detection...")
    
    try:
        # Use HU thresholding instead of TotalSegmentator
        l3_idx = find_l3_slice_by_hu_threshold(ct_path)
        
        if l3_idx is not None:
            print(f"  ✅ L3 slice detected: {l3_idx}")
        
        return l3_idx
        
    except Exception as e:
        print(f"  ❌ Stage 1 failed: {e}")
        return None


def stage2_segment_psoas(hu_slice: np.ndarray, vertebra_center_x: int, output_dir: str) -> bool:
    """
    STAGE 2: Segment psoas using HU thresholding and anatomical position.
    
    Args:
        hu_slice: HU values for L3 slice
        vertebra_center_x: X coordinate of vertebra center
        output_dir: Output directory for psoas masks
        
    Returns:
        Success status
    """
    print("  🎯 STAGE 2: Segmenting psoas using HU thresholding...")
    
    try:
        # Use HU-based psoas detection
        left_psoas, right_psoas = find_psoas_by_hu_and_position(
            hu_slice, vertebra_center_x, hu_slice.shape[0], hu_slice.shape[1]
        )
        
        # Save psoas masks as NIfTI files
        psoas_left_path = Path(output_dir) / "psoas_major_left.nii.gz"
        psoas_right_path = Path(output_dir) / "psoas_major_right.nii.gz"
        
        # Create NIfTI images (single slice)
        left_nib = nib.Nifti1Image(left_psoas[np.newaxis, :, :], np.eye(4))
        right_nib = nib.Nifti1Image(right_psoas[np.newaxis, :, :], np.eye(4))
        
        nib.save(left_nib, str(psoas_left_path))
        nib.save(right_nib, str(psoas_right_path))
        
        left_pixels = left_psoas.sum()
        right_pixels = right_psoas.sum()
        
        if left_pixels > 0 and right_pixels > 0:
            print(f"  ✅ Psoas masks generated: left={left_pixels}px, right={right_pixels}px")
            return True
        else:
            print(f"  ⚠️  Psoas masks incomplete: left={left_pixels}px, right={right_pixels}px")
            return False
        
    except Exception as e:
        print(f"  ❌ Stage 2 failed: {e}")
        return False


def process_case_two_stage(ct_path: str, case_id: str, output_dir: str) -> bool:
    """
    Process a single case using two-stage approach.
    
    Args:
        ct_path: Path to CT NII.GZ file
        case_id: Case identifier (e.g., "amos_0001")
        output_dir: Output directory for final masks
        
    Returns:
        Success status
    """
    print(f"\n{'='*60}")
    print(f"📌 Processing: {case_id} (TWO-STAGE)")
    print(f"{'='*60}")
    
    with tempfile.TemporaryDirectory() as temp_root:
        stage1_dir = Path(temp_root) / "stage1"
        stage2_dir = Path(temp_root) / "stage2"
        stage1_dir.mkdir()
        stage2_dir.mkdir()
        
        # STAGE 1: Find L3
        l3_idx = stage1_find_l3(ct_path, str(stage1_dir))
        
        if l3_idx is None:
            print("  ❌ L3 detection failed")
            return False
        
        # Extract L3 slice
        l3_slice_path = str(stage2_dir / "l3_slice.nii.gz")
        if not extract_single_slice(ct_path, l3_idx, l3_slice_path):
            return False
        
        # Load HU slice & spacing from saved single-slice NIfTI
        l3_nib = nib.load(l3_slice_path)
        l3_data = l3_nib.get_fdata()
        print(f"  🔍 L3 data shape: {l3_data.shape}, ndim: {l3_data.ndim}")
        # Extract the 2D slice from 3D volume (shape should be H x W x 1)
        hu_slice = l3_data[:, :, 0]  # Take the first (and only) slice
        print(f"  🔍 HU slice shape: {hu_slice.shape}")
        zooms = l3_nib.header.get_zooms()
        pixel_area = float(zooms[1] * zooms[2])
        print(f"  📐 Pixel area: {pixel_area:.3f} mm²")
        
        # Find vertebra center for psoas positioning
        # Use slice center as vertebra approximation (more reliable than moments)
        vertebra_center_x = hu_slice.shape[1] // 2
        print(f"  📍 Vertebra center X: {vertebra_center_x} (slice center)")
        
        # STAGE 2: Segment psoas using HU thresholding
        stage2_output = str(stage2_dir / "psoas_output")
        Path(stage2_output).mkdir(exist_ok=True)
        
        if not stage2_segment_psoas(hu_slice, vertebra_center_x, stage2_output):
            return False
        
        # STAGE 2: Segment psoas using HU thresholding
        stage2_output = str(stage2_dir / "psoas_output")
        Path(stage2_output).mkdir(exist_ok=True)
        
        if not stage2_segment_psoas(hu_slice, vertebra_center_x, stage2_output):
            return False
        
        # Load psoas masks (single slice)
        psoas_left_path = Path(stage2_output) / "psoas_major_left.nii.gz"
        psoas_right_path = Path(stage2_output) / "psoas_major_right.nii.gz"
        psoas_left = load_mask_slice(psoas_left_path, 0) if psoas_left_path.exists() else None
        psoas_right = load_mask_slice(psoas_right_path, 0) if psoas_right_path.exists() else None
        if psoas_left is None or psoas_right is None:
            print("  ⚠️  Psoas masks missing after Stage 2")
            return False
        
        # Build fat & muscle masks
        fat_mask = build_fat_mask(stage1_dir, l3_idx, hu_slice)
        muscle_mask = build_muscle_mask(hu_slice)
        
        # Area computations (mm²)
        pma_px = int((psoas_left + psoas_right).sum())
        pma_mm2 = float(pma_px * pixel_area)
        fat_mm2 = float(fat_mask.sum() * pixel_area)
        muscle_mm2 = float(muscle_mask.sum() * pixel_area)
        print(f"  📏 Areas → PMA: {pma_mm2:.1f} mm² | VFA: {fat_mm2:.1f} mm² | SMA: {muscle_mm2:.1f} mm²")
        
        # Copy results to final output directory
        case_output = Path(output_dir) / case_id
        case_output.mkdir(parents=True, exist_ok=True)
        
        # Copy psoas masks
        for psoas_file in Path(stage2_output).rglob("psoas_major_*.nii.gz"):
            shutil.copy(psoas_file, case_output / psoas_file.name)
            print(f"  ✅ Saved: {psoas_file.name}")

        # Save additional masks as PNG for QA
        # Clip HU to display range [-150, 250] before converting to 0-255
        hu_display = np.clip(hu_slice, -150, 250)
        cv2.imwrite(str(case_output / "hu_slice.png"), np.clip((hu_display + 150) / 400 * 255, 0, 255).astype(np.uint8))
        cv2.imwrite(str(case_output / "fat_mask.png"), (fat_mask * 255).astype(np.uint8))
        cv2.imwrite(str(case_output / "muscle_mask.png"), (muscle_mask * 255).astype(np.uint8))
        cv2.imwrite(str(case_output / "psoas_mask.png"), ((psoas_left + psoas_right) * 255).astype(np.uint8))
        
        # Save arrays as .npy for training (expected by step2_train_unet.py)
        np.save(str(case_output / "hu_slice.npy"), hu_slice.astype(np.float32))
        np.save(str(case_output / "psoas_left.npy"), psoas_left.astype(np.uint8))
        np.save(str(case_output / "psoas_right.npy"), psoas_right.astype(np.uint8))
        np.save(str(case_output / "vat.npy"), fat_mask.astype(np.uint8))  # Visceral adipose tissue
        
        # Save metadata
        metadata = {
            "case_id": case_id,
            "l3_slice_idx": int(l3_idx),
            "pixel_area_mm2": pixel_area,
            "PMA_mm2": pma_mm2,
            "VFA_mm2": fat_mm2,
            "SMA_mm2": muscle_mm2,
            "status": "success"
        }
        with open(case_output / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✅ {case_id} completed!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Two-stage L3 psoas segmentation")
    parser.add_argument("--input-data", required=True, help="Input CT directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of cases")
    args = parser.parse_args()
    
    input_dir = Path(args.input_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all NIFTI files
    ct_files = sorted(input_dir.glob("*.nii.gz"))
    
    if args.max_cases:
        ct_files = ct_files[:args.max_cases]
        print(f"⚠️  Limiting to {args.max_cases} cases")
    
    print(f"📊 Found {len(ct_files)} CT files")
    
    # Process each case
    success_count = 0
    failed_count = 0
    
    for ct_file in ct_files:
        case_id = ct_file.stem.replace(".nii", "")
        
        try:
            if process_case_two_stage(str(ct_file), case_id, str(output_dir)):
                success_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            failed_count += 1
    
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY: {success_count} success, {failed_count} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
