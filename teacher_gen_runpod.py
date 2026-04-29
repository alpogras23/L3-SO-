#!/usr/bin/env python3
"""
Simplified teacher generation for RunPod
"""
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    from totalsegmentator.python_api import totalsegmentator
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files = []
    # AMOS22 structure: look for CT scans
    # Check for standard AMOS naming (amos_XXXX.nii.gz) in imagesTr folder
    images_tr = input_path / "imagesTr"
    if images_tr.exists():
        for ct_file in images_tr.glob("amos_*.nii.gz"):
            files.append(ct_file)
            print(f"Found: {ct_file.name}")
    else:
        # Fallback: look for CT scans in subdirectories
        for subdir in input_path.iterdir():
            if subdir.is_dir() and not subdir.name.endswith('_labels'):
                # Try both .nii.gz and .nii extensions
                ct_file = subdir / f"{subdir.name}.nii.gz"
                if not ct_file.exists():
                    ct_file = subdir / f"{subdir.name}.nii"
                print(f"Checking: {ct_file} - exists: {ct_file.exists()}")
                if ct_file.exists():
                    files.append(ct_file)
    
    files = sorted(files)
    print(f"Found {len(files)} files to process")
    if args.limit:
        files = files[:args.limit]
    
    print(f"Processing {len(files)} files...")
    
    for ct_file in tqdm(files):
        case_name = ct_file.stem.replace(".nii", "")
        case_out = output_path / case_name
        case_out.mkdir(exist_ok=True)
        
        try:
            totalsegmentator(
                input=str(ct_file),
                output=str(case_out),
                task="abdominal_muscles",
                fast=False
            )
            print(f"✓ {case_name}")
        except Exception as e:
            print(f"✗ {case_name}: {e}")
    
    print("Done!")

if __name__ == "__main__":
    main()
