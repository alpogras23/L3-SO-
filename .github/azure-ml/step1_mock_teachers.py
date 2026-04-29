#!/usr/bin/env python3
"""
Simplified Step 1: Generate mock teacher labels from manifest
For testing purposes when actual DICOM files aren't available locally
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np

def create_mock_teachers(output_dir, num_cases=8):
    """Create mock teacher labels for testing"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[MOCK] Creating {num_cases} mock teacher labels...")
    
    results = []
    for case_id in range(1, num_cases + 1):
        case_dir = output_dir / f"amos_{case_id:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock HU slice (256x256)
        hu_slice = np.random.normal(0, 100, (256, 256)).astype(np.float32)
        hu_slice = np.clip(hu_slice, -150, 200)
        
        # Mock psoas masks
        psoas_left = np.zeros((256, 256), dtype=np.float32)
        psoas_left[100:150, 80:120] = 1.0
        
        psoas_right = np.zeros((256, 256), dtype=np.float32)
        psoas_right[100:150, 136:176] = 1.0
        
        # Mock VAT mask
        vat = np.zeros((256, 256), dtype=np.float32)
        vat[40:200, 40:216] = 1.0
        vat[80:176, 80:176] = 0.0  # Inner abdomen exclusion
        
        # Save
        np.save(case_dir / "hu_slice.npy", hu_slice)
        np.save(case_dir / "psoas_left.npy", psoas_left)
        np.save(case_dir / "psoas_right.npy", psoas_right)
        np.save(case_dir / "vat.npy", vat)
        
        metadata = {
            "case_id": f"amos_{case_id:04d}",
            "hu_range": [float(hu_slice.min()), float(hu_slice.max())],
            "psoas_left_pixels": int(psoas_left.sum()),
            "psoas_right_pixels": int(psoas_right.sum()),
            "vat_pixels": int(vat.sum()),
            "pixel_spacing_mm": 1.5,
        }
        
        with open(case_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        results.append(metadata)
        print(f"  ✓ Created: amos_{case_id:04d}")
    
    # Summary
    summary = {
        "num_cases": num_cases,
        "cases": results,
        "status": "success"
    }
    
    with open(output_dir.parent / "teacher_generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Mock teachers created: {output_dir}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mock teacher labels")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--num-cases", type=int, default=8, help="Number of mock cases")
    
    args = parser.parse_args()
    
    try:
        create_mock_teachers(args.output_dir, args.num_cases)
        print("\n✅ Success!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
