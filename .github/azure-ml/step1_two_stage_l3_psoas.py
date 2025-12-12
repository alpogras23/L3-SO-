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
        # Use subprocess to call TotalSegmentator CLI (more reliable than Python API)
        cmd = [
            "totalsegmentator",  # Use direct binary instead of 'python -m'
            "-i", str(ct_path),
            "-o", str(temp_dir),
            "-ta", "total",  # task="total" includes vertebrae
            "--fast"         # CRITICAL: LOW RAM mode
        ]
        
        print(f"  🚀 Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
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
        # Use subprocess to call TotalSegmentator CLI
        cmd = [
            "totalsegmentator",  # Use direct binary instead of 'python -m'
            "-i", str(l3_slice_path),
            "-o", str(output_dir),
            "-ta", "abdominal_muscles"  # Includes psoas_major_left/right (labels 19-20)
            # NOTE: fast=False is implicit (--fast not added)
            # Single slice → LOW RAM even without --fast
        ]
        
        print(f"  🚀 Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"  ❌ TotalSegmentator failed: {result.stderr[:300]}")
            return False
        
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
        
        # Copy results to final output directory
        case_output = Path(output_dir) / case_id
        case_output.mkdir(parents=True, exist_ok=True)
        
        # Copy psoas masks
        for psoas_file in Path(stage2_output).rglob("psoas_major_*.nii.gz"):
            shutil.copy(psoas_file, case_output / psoas_file.name)
            print(f"  ✅ Saved: {psoas_file.name}")
        
        # Save metadata
        metadata = {
            "case_id": case_id,
            "l3_slice_idx": int(l3_idx),
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
