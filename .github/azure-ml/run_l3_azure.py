#!/usr/bin/env python3
"""
Azure ML L3 VFA/PMA Processing Script
Processes AMOS22 NIfTI files and outputs VFA/PMA measurements
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk

def load_nifti(path):
    """Load NIfTI file and return numpy array + metadata"""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    return arr, spacing

def basic_vfa_pma_calculation(hu_slice, spacing):
    """
    Simplified VFA/PMA calculation for Azure ML
    Full pipeline will be integrated after validation
    """
    # Simple thresholding for demonstration
    # VFA: Fat tissue (-190 to -30 HU)
    vfa_mask = (hu_slice >= -190) & (hu_slice <= -30)
    
    # PMA: Muscle tissue near vertebra (-29 to 150 HU)
    # This is simplified - full version uses psoas detection
    pma_mask = (hu_slice >= -29) & (hu_slice <= 150)
    
    # Calculate areas
    pixel_area_mm2 = spacing[0] * spacing[1]
    vfa_mm2 = float(np.sum(vfa_mask) * pixel_area_mm2)
    pma_mm2 = float(np.sum(pma_mask) * pixel_area_mm2)
    
    return vfa_mm2, pma_mm2

def process_case(ct_path, output_dir):
    """Process single AMOS22 case"""
    case_id = ct_path.stem
    print(f"Processing case: {case_id}")
    
    try:
        # Load CT volume
        hu_volume, spacing = load_nifti(ct_path)
        print(f"  Volume shape: {hu_volume.shape}, spacing: {spacing}")
        
        # Select middle slice (L3 detection will be added)
        mid_slice = hu_volume.shape[0] // 2
        hu_slice = hu_volume[mid_slice]
        print(f"  Selected slice: {mid_slice}")
        
        # Calculate VFA/PMA
        vfa_mm2, pma_mm2 = basic_vfa_pma_calculation(hu_slice, spacing)
        vfa_cm2 = vfa_mm2 / 100.0
        pma_cm2 = pma_mm2 / 100.0
        
        print(f"  VFA: {vfa_cm2:.2f} cm²")
        print(f"  PMA: {pma_cm2:.2f} cm²")
        
        # Save results
        results = {
            "case_id": case_id,
            "slice_index": int(mid_slice),
            "vfa_mm2": float(vfa_mm2),
            "pma_mm2": float(pma_mm2),
            "vfa_cm2": float(vfa_cm2),
            "pma_cm2": float(pma_cm2),
            "status": "success"
        }
        
        output_path = output_dir / f"{case_id}_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
        
    except Exception as e:
        print(f"  ERROR: {e}")
        error_result = {
            "case_id": case_id,
            "status": "error",
            "error": str(e)
        }
        output_path = output_dir / f"{case_id}_error.json"
        with open(output_path, 'w') as f:
            json.dump(error_result, f, indent=2)
        return error_result

def main():
    parser = argparse.ArgumentParser(description='L3 VFA/PMA Analysis - Azure ML')
    parser.add_argument('--input-data', required=True, help='Input data directory (AMOS22 NIfTI files)')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    args = parser.parse_args()
    
    input_dir = Path(args.input_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("L3 VFA/PMA Analysis - Azure ML")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Find all CT NIfTI files
    ct_files = list(input_dir.glob("*_0000.nii.gz"))
    print(f"Found {len(ct_files)} CT files")
    print()
    
    if not ct_files:
        print("ERROR: No CT files found!")
        print(f"Searched for: {input_dir}/*_0000.nii.gz")
        print(f"Directory contents: {list(input_dir.glob('*'))[:10]}")
        sys.exit(1)
    
    # Process all cases
    all_results = []
    for ct_path in sorted(ct_files):
        result = process_case(ct_path, output_dir)
        all_results.append(result)
        print()
    
    # Save summary
    summary = {
        "total_cases": len(all_results),
        "successful": sum(1 for r in all_results if r.get("status") == "success"),
        "failed": sum(1 for r in all_results if r.get("status") == "error"),
        "results": all_results
    }
    
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total cases: {summary['total_cases']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Summary saved to: {summary_path}")
    
    return 0 if summary['failed'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
