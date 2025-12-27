#!/usr/bin/env python3
"""
Simplified teacher generation script for RunPod
Generates teacher labels using TotalSegmentator for psoas segmentation
"""

import os
import sys
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
from tqdm import tqdm

def run_totalsegmentator(input_file, output_dir):
    """Run TotalSegmentator on a single CT file"""
    try:
        from totalsegmentator.python_api import totalsegmentator
        
        print(f"Processing: {input_file}")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run TotalSegmentator with abdominal_muscles task for psoas
        totalsegmentator(
            input=str(input_file),
            output=str(output_path),
            task="abdominal_muscles",
            fast=False,
            quiet=False
        )
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate teacher labels with TotalSegmentator")
    parser.add_argument("--input-dir", required=True, help="Input directory with .nii.gz files")
    parser.add_argument("--output-dir", required=True, help="Output directory for labels")
    parser.add_argument("--limit", type=int, help="Limit number of files to process")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .nii.gz files
    files = sorted(input_path.glob("*.nii.gz"))
    
    if args.limit:
        files = files[:args.limit]
    
    print(f"Found {len(files)} files to process")
    
    success_count = 0
    for ct_file in tqdm(files, desc="Processing CT scans"):
        case_name = ct_file.stem.replace(".nii", "")
        case_output_dir = output_path / case_name
        
        if run_totalsegmentator(ct_file, case_output_dir):
            success_count += 1
    
    print(f"\nComplete! Processed {success_count}/{len(files)} files")
    print(f"Output directory: {output_path}")

if __name__ == "__main__":
    main()
