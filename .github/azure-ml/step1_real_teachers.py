#!/usr/bin/env python3
"""
Step 1: Real Teacher Generation from AMOS22 NII.GZ files using TotalSegmentator
Reads from /Users/alperenogras/Desktop/amos22/imagesTr/ and generates L3 teacher labels
"""
import sys
import json
import argparse
import subprocess
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm

def run_totalsegmentator(ct_path, output_dir, fast=True):
    """Run TotalSegmentator on single CT volume"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    marker = output_dir / "TS_COMPLETE"
    if marker.exists():
        print(f"  ✓ Already processed: {ct_path.stem}")
        return True
    
    print(f"  Running TotalSegmentator: {ct_path.stem}")
    
    try:
        cmd = [
            sys.executable, "-m", "totalsegmentator",
            "-i", str(ct_path),
            "-o", str(output_dir),
            "-ta", "total",
        ]
        
        if fast:
            cmd.append("--fast")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            marker.touch()
            print(f"  ✓ Success: {ct_path.stem}")
            return True
        else:
            print(f"  ✗ Failed: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:200]}")
        return False

def pick_l3_slice(ts_output_dir):
    """Pick L3 vertebra slice from TS output"""
    ts_dir = Path(ts_output_dir)
    
    # Look for vertebrae masks
    vertebra_files = [
        ts_dir / "vertebrae_L3.nii.gz",
        ts_dir / "L3.nii.gz",
    ]
    
    for vfile in vertebra_files:
        if vfile.exists():
            try:
                data = nib.load(vfile).get_fdata()
                # Find slice with most vertebra voxels
                sums = np.sum(data > 0, axis=(0, 1))
                if sums.max() > 0:
                    z_idx = int(np.argmax(sums))
                    return z_idx, data
            except:
                pass
    
    # Fallback: middle slice
    return None, None

def extract_teacher_labels(ct_path, ts_output_dir, output_dir):
    """Extract 2D teacher labels at L3 from CT and TS masks"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load CT
        ct_data = nib.load(ct_path).get_fdata()
        ct_affine = nib.load(ct_path).affine
        pixel_spacing = ct_affine[0, 0]  # Assuming isotropic
        
        # Find L3 slice
        z_idx, _ = pick_l3_slice(ts_output_dir)
        
        if z_idx is None:
            print(f"  Could not detect L3, using middle slice")
            z_idx = ct_data.shape[2] // 2
        
        # Extract 2D slice
        hu_slice = ct_data[:, :, z_idx].astype(np.float32)
        
        # Load masks (if available)
        psoas_left = np.zeros_like(hu_slice)
        psoas_right = np.zeros_like(hu_slice)
        vat = np.zeros_like(hu_slice)
        
        ts_dir = Path(ts_output_dir)
        
        # Try to load psoas masks
        for mask_file in [ts_dir / "psoas_major_left.nii.gz", ts_dir / "psoas_left.nii.gz"]:
            if mask_file.exists():
                mask_data = nib.load(mask_file).get_fdata()
                psoas_left = mask_data[:, :, z_idx].astype(np.float32) > 0
                break
        
        for mask_file in [ts_dir / "psoas_major_right.nii.gz", ts_dir / "psoas_right.nii.gz"]:
            if mask_file.exists():
                mask_data = nib.load(mask_file).get_fdata()
                psoas_right = mask_data[:, :, z_idx].astype(np.float32) > 0
                break
        
        # Try to load visceral fat (from various possible labels)
        for mask_file in [ts_dir / "visceral_fat.nii.gz", ts_dir / "torso_fat.nii.gz"]:
            if mask_file.exists():
                mask_data = nib.load(mask_file).get_fdata()
                vat = mask_data[:, :, z_idx].astype(np.float32) > 0
                break
        
        # Resize to 256x256 if needed
        from scipy import ndimage
        original_shape = hu_slice.shape
        
        if original_shape != (256, 256):
            scale = 256.0 / original_shape[0]
            hu_slice = ndimage.zoom(hu_slice, scale, order=1)
            psoas_left = ndimage.zoom(psoas_left.astype(float), scale, order=0) > 0.5
            psoas_right = ndimage.zoom(psoas_right.astype(float), scale, order=0) > 0.5
            vat = ndimage.zoom(vat.astype(float), scale, order=0) > 0.5
        
        # Save
        np.save(output_dir / "hu_slice.npy", hu_slice)
        np.save(output_dir / "psoas_left.npy", psoas_left.astype(np.float32))
        np.save(output_dir / "psoas_right.npy", psoas_right.astype(np.float32))
        np.save(output_dir / "vat.npy", vat.astype(np.float32))
        
        # Metadata
        metadata = {
            "case_id": ct_path.stem,
            "z_index": int(z_idx),
            "hu_range": [float(hu_slice.min()), float(hu_slice.max())],
            "psoas_left_pixels": int(psoas_left.sum()),
            "psoas_right_pixels": int(psoas_right.sum()),
            "vat_pixels": int(vat.sum()),
            "pixel_spacing_mm": float(abs(pixel_spacing)),
        }
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Extracted: {ct_path.stem} at z={z_idx}")
        return True, metadata
        
    except Exception as e:
        print(f"  ✗ Error extracting: {str(e)[:200]}")
        return False, None

def main():
    parser = argparse.ArgumentParser(description="Generate AMOS22 teacher labels with TotalSegmentator")
    parser.add_argument("--input-data", default="/Users/alperenogras/Desktop/amos22/imagesTr", help="Input CT directory")
    parser.add_argument("--output-dir", required=True, help="Output teacher labels directory")
    parser.add_argument("--fast", action="store_true", default=True, help="Use fast mode")
    parser.add_argument("--ts-dir", default="/tmp/ts_output", help="Temporary TS output directory")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all AMOS CT files
    ct_files = sorted(input_dir.glob("amos_*.nii.gz"))
    
    if not ct_files:
        print(f"❌ No AMOS NII.GZ files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(ct_files)} AMOS files")
    print()
    
    results = []
    success_count = 0
    
    for ct_path in tqdm(ct_files, desc="Processing"):
        case_id = ct_path.stem
        case_ts_dir = Path(args.ts_dir) / case_id
        case_output_dir = output_dir / case_id
        
        # Step 1: Run TotalSegmentator
        if run_totalsegmentator(ct_path, case_ts_dir, fast=args.fast):
            # Step 2: Extract teacher labels
            success, meta = extract_teacher_labels(ct_path, case_ts_dir, case_output_dir)
            if success:
                results.append(meta)
                success_count += 1
    
    # Summary
    summary = {
        "num_total": len(ct_files),
        "num_success": success_count,
        "cases": results,
        "status": "success" if success_count > 0 else "failed"
    }
    
    with open(output_dir.parent / "teacher_generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print()
    print(f"✅ Teacher generation completed: {success_count}/{len(ct_files)} cases")
    
    if success_count > 0:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
