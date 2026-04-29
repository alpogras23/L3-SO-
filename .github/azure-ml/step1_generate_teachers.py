#!/usr/bin/env python3
"""
Azure ML Full Pipeline: AMOS22 + TotalSegmentator + U-Net Training
Step 1: TotalSegmentator Teacher Generation
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
        print(f"  Already processed: {ct_path.stem}")
        return True
    
    print(f"  Running TotalSegmentator: {ct_path.stem}")
    
    try:
        # Run for vertebrae + abdominal structures
        cmd = [
            sys.executable, "-m", "totalsegmentator",
            "-i", str(ct_path),
            "-o", str(output_dir),
            "-ta", "total",
            "--roi_subset", "vertebrae_L3",
            "--roi_subset", "vertebrae_L4",
            "--roi_subset", "vertebrae_L5",
            "--roi_subset", "psoas_major",
            "--roi_subset", "subcutaneous_fat",
            "--roi_subset", "torso_fat"
        ]
        
        if fast:
            cmd.append("--fast")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            marker.touch()
            print(f"  ✓ Success: {ct_path.stem}")
            return True
        else:
            print(f"  ✗ Failed: {ct_path.stem}")
            print(f"    Error: {result.stderr[:200]}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {ct_path.stem} - {e}")
        return False

def pick_l3_slice(ts_output_dir):
    """Pick L3-like slice from vertebrae masks"""
    # Try vertebrae L3 first
    vertebrae_files = [
        ts_output_dir / "vertebrae_L3.nii.gz",
        ts_output_dir / "vertebrae_L4.nii.gz",
        ts_output_dir / "vertebrae_lumbar.nii.gz"
    ]
    
    for vf in vertebrae_files:
        if vf.exists():
            img = nib.load(str(vf))
            vol = img.get_fdata()
            
            # Find slice with maximum vertebra area
            areas = vol.reshape(vol.shape[0], -1).sum(axis=1)
            if areas.max() > 0:
                z_index = int(areas.argmax())
                print(f"    L3 slice detected at z={z_index} from {vf.name}")
                return z_index
    
    # Fallback: middle slice
    print("    Warning: Using middle slice (L3 not detected)")
    return None

def extract_teacher_labels(ct_path, ts_output_dir, output_dir):
    """Extract 2D teacher labels from 3D TotalSegmentator masks"""
    case_id = ct_path.stem
    
    # Load CT
    ct_img = nib.load(str(ct_path))
    ct_vol = ct_img.get_fdata().astype(np.float32)
    spacing = ct_img.header.get_zooms()
    
    # Pick L3 slice
    z_index = pick_l3_slice(ts_output_dir)
    if z_index is None:
        z_index = ct_vol.shape[0] // 2
    
    hu_slice = ct_vol[z_index]
    
    # Load TS masks for this slice
    labels = {}
    
    # Psoas (left + right)
    psoas_left_path = ts_output_dir / "psoas_major_left.nii.gz"
    psoas_right_path = ts_output_dir / "psoas_major_right.nii.gz"
    
    if psoas_left_path.exists():
        psoas_img = nib.load(str(psoas_left_path))
        labels['psoas_left'] = (psoas_img.get_fdata()[z_index] > 0).astype(np.uint8)
    else:
        labels['psoas_left'] = np.zeros_like(hu_slice, dtype=np.uint8)
    
    if psoas_right_path.exists():
        psoas_img = nib.load(str(psoas_right_path))
        labels['psoas_right'] = (psoas_img.get_fdata()[z_index] > 0).astype(np.uint8)
    else:
        labels['psoas_right'] = np.zeros_like(hu_slice, dtype=np.uint8)
    
    # VAT (visceral fat - inside fascia)
    torso_fat_path = ts_output_dir / "torso_fat.nii.gz"
    if torso_fat_path.exists():
        fat_img = nib.load(str(torso_fat_path))
        fat_mask = (fat_img.get_fdata()[z_index] > 0).astype(np.uint8)
        # Simplified: all torso fat as VAT (proper fascia segmentation needed)
        labels['vat'] = fat_mask
    else:
        # HU-based fallback
        labels['vat'] = ((hu_slice >= -190) & (hu_slice <= -30)).astype(np.uint8)
    
    # SAT (subcutaneous fat)
    sat_path = ts_output_dir / "subcutaneous_fat.nii.gz"
    if sat_path.exists():
        sat_img = nib.load(str(sat_path))
        labels['sat'] = (sat_img.get_fdata()[z_index] > 0).astype(np.uint8)
    else:
        labels['sat'] = np.zeros_like(hu_slice, dtype=np.uint8)
    
    # Save 2D slices
    output_case_dir = output_dir / case_id
    output_case_dir.mkdir(parents=True, exist_ok=True)
    
    # Save HU slice
    np.save(output_case_dir / "hu_slice.npy", hu_slice)
    
    # Save labels
    for label_name, label_mask in labels.items():
        np.save(output_case_dir / f"{label_name}.npy", label_mask)
    
    # Save metadata
    metadata = {
        "case_id": case_id,
        "z_index": int(z_index),
        "spacing": [float(s) for s in spacing],
        "shape": list(hu_slice.shape)
    }
    
    with open(output_case_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description='TotalSegmentator Teacher Generation')
    parser.add_argument('--input-data', required=True, help='Input AMOS22 NIfTI directory')
    parser.add_argument('--output-dir', required=True, help='Output directory for teacher labels')
    parser.add_argument('--max-cases', type=int, default=None, help='Max cases to process (for testing)')
    parser.add_argument('--fast', action='store_true', help='Use TotalSegmentator fast mode')
    args = parser.parse_args()
    
    input_dir = Path(args.input_data)
    output_dir = Path(args.output_dir)
    
    ts_cache_dir = output_dir / "ts_raw"
    teacher_dir = output_dir / "teacher_labels"
    
    ts_cache_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("STEP 1: TotalSegmentator Teacher Generation")
    print("=" * 70)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Find CT files
    ct_files = list(input_dir.glob("*_0000.nii.gz"))
    if not ct_files:
        ct_files = list(input_dir.glob("*.nii.gz"))
    
    if args.max_cases:
        ct_files = ct_files[:args.max_cases]
    
    print(f"Found {len(ct_files)} CT files")
    print()
    
    # Process each case
    results = []
    for ct_path in tqdm(ct_files, desc="Processing cases"):
        case_id = ct_path.stem
        ts_out = ts_cache_dir / case_id
        
        # Run TotalSegmentator
        ts_success = run_totalsegmentator(ct_path, ts_out, fast=args.fast)
        
        if ts_success:
            # Extract 2D teacher labels
            try:
                metadata = extract_teacher_labels(ct_path, ts_out, teacher_dir)
                results.append({
                    "case_id": case_id,
                    "status": "success",
                    "metadata": metadata
                })
            except Exception as e:
                print(f"  Error extracting labels: {e}")
                results.append({
                    "case_id": case_id,
                    "status": "failed",
                    "error": str(e)
                })
        else:
            results.append({
                "case_id": case_id,
                "status": "failed",
                "error": "TotalSegmentator failed"
            })
    
    # Save summary
    summary = {
        "total_cases": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results
    }
    
    summary_path = output_dir / "teacher_generation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total: {summary['total_cases']}")
    print(f"Success: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Summary: {summary_path}")
    
    return 0 if summary['failed'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
