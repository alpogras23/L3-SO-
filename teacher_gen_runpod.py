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
    
    files = sorted(input_path.glob("*.nii.gz"))
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
