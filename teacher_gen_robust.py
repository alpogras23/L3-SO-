#!/usr/bin/env python3
"""
Robust teacher generation script for AMOS22 data.
Handles .nii and .nii.gz files.
Skips already processed cases.
"""
import os
import sys
import argparse
from pathlib import Path
import time
from tqdm import tqdm
import nibabel as nib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Root directory containing imagesTr")
    parser.add_argument("--output-dir", required=True, help="Output directory for masks")
    args = parser.parse_args()
    
    # Import here to avoid slow startup if just checking help
    print("Importing TotalSegmentator...")
    from totalsegmentator.python_api import totalsegmentator
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find files
    images_tr = input_path / "imagesTr"
    if not images_tr.exists():
        # Fallback to root if imagesTr not found
        images_tr = input_path
        
    print(f"Scanning for images in {images_tr}...")
    
    files = []
    # Look for both .nii and .nii.gz
    candidates = sorted(list(images_tr.glob("amos_*.nii")) + list(images_tr.glob("amos_*.nii.gz")))
    
    # Filter duplicates (prefer .nii.gz if both exist, though unlikely here)
    seen_stems = set()
    for f in candidates:
        # Handle double extension .nii.gz
        stem = f.name.replace(".nii.gz", "").replace(".nii", "")
        if stem not in seen_stems:
            files.append(f)
            seen_stems.add(stem)
            
    print(f"Found {len(files)} unique cases.")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for ct_file in tqdm(files, desc="Processing cases"):
        # Determine case name
        case_name = ct_file.name.replace(".nii.gz", "").replace(".nii", "")
        case_out_dir = output_path / case_name
        
        # Check if already processed
        # We check for a key file like 'skeletal_muscle.nii.gz' or just if the dir exists and is not empty
        if case_out_dir.exists():
            # Check for at least one expected output file to confirm it's not a failed run
            expected_file = case_out_dir / "skeletal_muscle.nii.gz" 
            # TotalSegmentator abdominal_muscles task produces specific files. 
            # Let's check if directory has content.
            if any(case_out_dir.iterdir()):
                print(f"Skipping {case_name} (already processed)")
                skipped_count += 1
                continue
        
        print(f"Processing {case_name} from {ct_file.name}...")
        case_out_dir.mkdir(exist_ok=True)
        
        start_time = time.time()
        try:
            # Run TotalSegmentator
            totalsegmentator(
                input=str(ct_file),
                output=str(case_out_dir),
                task="abdominal_muscles",
                fast=False,
                ml=True # Use ML mode if applicable/needed, but default is fine
            )
            duration = time.time() - start_time
            print(f"✓ Finished {case_name} in {duration:.1f}s")
            processed_count += 1
            
        except Exception as e:
            print(f"✗ Failed {case_name}: {e}")
            error_count += 1
            # Optional: remove empty directory if failed
            # if case_out_dir.exists() and not any(case_out_dir.iterdir()):
            #    case_out_dir.rmdir()

    print("-" * 50)
    print(f"Summary:")
    print(f"  Total found: {len(files)}")
    print(f"  Skipped:     {skipped_count}")
    print(f"  Processed:   {processed_count}")
    print(f"  Errors:      {error_count}")
    print("-" * 50)

if __name__ == "__main__":
    main()
