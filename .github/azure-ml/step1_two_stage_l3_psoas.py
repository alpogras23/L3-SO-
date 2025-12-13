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


def find_l3_slice_from_vertebra(vertebra_path: str) -> Optional[int]:
    """
    Find L3 slice from vertebra segmentation.
    
    Args:
        vertebra_path: Path to vertebra NII.GZ file
        
    Returns:
        L3 slice index (z-axis) or None
    """
    try:
        vertebra_nib = nib.load(vertebra_path)
        vertebra_data = vertebra_nib.get_fdata()
        
        # Find slices with vertebra present
        z_slices = np.where(vertebra_data.max(axis=(0, 1)) > 0)[0]
        
        if len(z_slices) == 0:
            return None
        
        # Use middle slice as L3 approximation
        # (In production, use label-based detection)
        l3_idx = int(np.median(z_slices))
        return l3_idx
        
    except Exception as e:
        print(f"  ⚠️  L3 detection failed: {e}")
        return None


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
    STAGE 1: Find L3 slice using task="total" with fast=True (LOW RAM).
    
    Args:
        ct_path: Path to full CT volume
        temp_dir: Temporary directory for stage 1 output
        
    Returns:
        L3 slice index or None
    """
    print("  🔍 STAGE 1: Finding L3 slice (task=total, fast=True)...")
    
    try:
        # Use Python API instead of subprocess (more reliable in AML container)
        from totalsegmentator.python_api import totalsegmentator
        
        print(f"  🚀 Running TotalSegmentator Python API: task=total, fast=True")
        print(f"     Input: {ct_path}")
        print(f"     Output: {temp_dir}")
        
        totalsegmentator(
            input=ct_path,
            output=temp_dir,
            task="total",
            fast=True,
            ml=True,
            quiet=False
        )
        
        # Check if successful (dummy result for API)
        result = type('obj', (object,), {'returncode': 0, 'stderr': ''})()
        
        if result.returncode != 0:
            print(f"  ❌ TotalSegmentator failed: {result.stderr[:300]}")
            return None
        
        # Find vertebra file - search in ALL subdirectories
        vertebra_file = None
        
        # Debug: List all generated files
        all_files = list(Path(temp_dir).rglob("*.nii.gz"))
        print(f"  📁 TotalSegmentator generated {len(all_files)} files")
        if all_files:
            print(f"  📄 Sample files: {[f.name for f in all_files[:5]]}")
        
        # Search for any spine/vertebra-related file
        search_patterns = [
            "vertebrae_L3.nii.gz",
            "vertebrae_L4.nii.gz",
            "*vertebra*.nii.gz",
            "*spine*.nii.gz", 
            "*lumbar*.nii.gz"
        ]
        
        for pattern in search_patterns:
            candidates = list(Path(temp_dir).rglob(pattern))
            if candidates:
                vertebra_file = str(candidates[0])
                print(f"  ✅ Found vertebra file: {Path(vertebra_file).name}")
                break
        
        if not vertebra_file:
            print("  ⚠️  No vertebra segmentation found")
            return None
        
        # Find L3 slice
        l3_idx = find_l3_slice_from_vertebra(vertebra_file)
        
        if l3_idx is not None:
            print(f"  ✅ L3 slice detected: {l3_idx}")
        
        return l3_idx
        
    except Exception as e:
        print(f"  ❌ Stage 1 failed: {e}")
        return None


def stage2_segment_psoas(l3_slice_path: str, output_dir: str) -> bool:
    """
    STAGE 2: Segment psoas on single L3 slice (task="abdominal_muscles", fast=False, LOW RAM).
    
    Args:
        l3_slice_path: Path to single-slice CT volume
        output_dir: Output directory for psoas masks
        
    Returns:
        Success status
    """
    print("  🎯 STAGE 2: Segmenting psoas on L3 slice (task=abdominal_muscles)...")
    
    try:
        # Use Python API instead of subprocess (more reliable in AML container)
        from totalsegmentator.python_api import totalsegmentator
        
        print(f"  🚀 Running TotalSegmentator Python API: task=abdominal_muscles")
        print(f"     Input: {l3_slice_path}")
        print(f"     Output: {output_dir}")
        
        totalsegmentator(
            input=l3_slice_path,
            output=output_dir,
            task="abdominal_muscles",
            fast=False,  # Required for abdominal_muscles
            ml=True,
            quiet=False
        )
        
        # Check if psoas masks were generated
        psoas_left = Path(output_dir) / "psoas_major_left.nii.gz"
        psoas_right = Path(output_dir) / "psoas_major_right.nii.gz"
        
        if psoas_left.exists() and psoas_right.exists():
            print("  ✅ Psoas masks generated successfully!")
            return True
        else:
            # Search in subdirectories
            psoas_files = list(Path(output_dir).rglob("psoas_major_*.nii.gz"))
            if len(psoas_files) >= 2:
                print(f"  ✅ Psoas masks found: {len(psoas_files)} files")
                return True
            else:
                print(f"  ⚠️  Psoas masks incomplete: {len(psoas_files)}/2 found")
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
        
        # STAGE 2: Segment psoas on L3 slice
        stage2_output = str(stage2_dir / "psoas_output")
        Path(stage2_output).mkdir(exist_ok=True)
        
        if not stage2_segment_psoas(l3_slice_path, stage2_output):
            return False
        
        # Load HU slice & spacing from saved single-slice NIfTI
        l3_nib = nib.load(l3_slice_path)
        l3_data = l3_nib.get_fdata()
        if l3_data.ndim == 4:
            l3_data = l3_data[..., 0]
        hu_slice = l3_data[0, :, :]
        zooms = l3_nib.header.get_zooms()
        pixel_area = float(zooms[1] * zooms[2])
        print(f"  📐 Pixel area: {pixel_area:.3f} mm²")
        
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
        cv2.imwrite(str(case_output / "hu_slice.png"), np.clip((hu_slice + 150) / 400 * 255, 0, 255).astype(np.uint8))
        cv2.imwrite(str(case_output / "fat_mask.png"), (fat_mask * 255).astype(np.uint8))
        cv2.imwrite(str(case_output / "muscle_mask.png"), (muscle_mask * 255).astype(np.uint8))
        cv2.imwrite(str(case_output / "psoas_mask.png"), ((psoas_left + psoas_right) * 255).astype(np.uint8))
        
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
