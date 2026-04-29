#!/usr/bin/env python3
"""
Batch Process DICOMNET Dataset for L3 VFA/PMA Analysis

Bu script DICOMNET klasöründeki tüm DICOM dosyalarını işler.
"""

import os
import subprocess
import argparse
from pathlib import Path

def batch_process_dicomnet(dicomnet_path: str, output_dir: str, max_patients: int = None):
    """
    Batch process all DICOM files in DICOMNET dataset.

    Args:
        dicomnet_path: Path to DICOMNET folder
        output_dir: Output directory
        max_patients: Maximum number of patients to process (None for all)
    """
    dicomnet_path = Path(dicomnet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all patient directories
    patient_dirs = [d for d in dicomnet_path.iterdir() if d.is_dir()]
    patient_dirs.sort()

    if max_patients:
        patient_dirs = patient_dirs[:max_patients]

    print(f"Found {len(patient_dirs)} patient directories")

    processed_count = 0
    successful_count = 0

    for patient_dir in patient_dirs:
        print(f"\n📁 Processing patient: {patient_dir.name}")

        # Get DICOM files in this patient directory
        dicom_files = list(patient_dir.glob("*.dcm"))
        dicom_files.sort()

        if not dicom_files:
            print(f"  ⚠️  No DICOM files found in {patient_dir.name}")
            continue

        # Process first DICOM file from each patient
        dicom_file = dicom_files[0]
        processed_count += 1

        print(f"  📄 Processing: {dicom_file.name}")

        # Create patient-specific output directory
        patient_output_dir = output_dir / patient_dir.name.replace(' ', '_').replace('.', '_')
        patient_output_dir.mkdir(exist_ok=True)

        try:
            # Run Comp2Comp calculator
            cmd = [
                "python", "comp2comp_enhanced_l3.py",
                "--input", str(dicom_file),
                "--output", str(patient_output_dir)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

            if result.returncode == 0:
                print(f"  ✅ Success: {dicom_file.name}")
                successful_count += 1

                # Print results summary
                results_file = patient_output_dir / f"{dicom_file.stem}_results.json"
                if results_file.exists():
                    try:
                        import json
                        with open(results_file, 'r') as f:
                            data = json.load(f)
                        print(f"     PMA: {data['pma_mm2']:.1f} mm², VFA: {data['vfa_mm2']:.1f} mm²")
                    except:
                        pass
            else:
                print(f"  ❌ Failed: {dicom_file.name}")
                print(f"     Error: {result.stderr[:200]}...")

        except Exception as e:
            print(f"  ❌ Exception: {dicom_file.name} - {e}")

    print("\n📊 Batch Processing Summary")
    print(f"   Total patients found: {len(patient_dirs)}")
    print(f"   Patients processed: {processed_count}")
    print(f"   Successful: {successful_count}")
    print(f"   Failed: {processed_count - successful_count}")
    print(f"   Success rate: {successful_count/processed_count*100:.1f}%" if processed_count > 0 else "   Success rate: N/A")

def main():
    parser = argparse.ArgumentParser(description="Batch Process DICOMNET Dataset")
    parser.add_argument("--dicomnet-path", default="/Users/alperenogras/Desktop/DICOMNET",
                       help="Path to DICOMNET folder")
    parser.add_argument("--output-dir", default="./dicomnet_full_analysis",
                       help="Output directory")
    parser.add_argument("--max-patients", type=int, default=None,
                       help="Maximum number of patients to process")

    args = parser.parse_args()

    print("🚀 Starting DICOMNET Batch Processing")
    print(f"   DICOMNET Path: {args.dicomnet_path}")
    print(f"   Output Directory: {args.output_dir}")
    print(f"   Max Patients: {args.max_patients or 'All'}")

    batch_process_dicomnet(args.dicomnet_path, args.output_dir, args.max_patients)

if __name__ == "__main__":
    main()