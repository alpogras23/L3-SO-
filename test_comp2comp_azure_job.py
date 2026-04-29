#!/usr/bin/env python3
"""
Test Comp2Comp Azure ML Inference Job
Prepares AMOS data and runs the Comp2Comp inference pipeline
"""

import os
import sys
import json
import argparse
from pathlib import Path
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Test Comp2Comp Azure ML Inference")
    parser.add_argument("--amos-root", required=True, help="Path to AMOS dataset root")
    parser.add_argument("--output-dir", default="./comp2comp_azure_test", help="Output directory")
    parser.add_argument("--max-cases", type=int, default=3, help="Maximum number of test cases")
    parser.add_argument("--submit-job", action="store_true", help="Submit Azure ML job")

    args = parser.parse_args()

    amos_root = Path(args.amos_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check AMOS data
    images_dir = amos_root / "imagesTr"
    if not images_dir.exists():
        print(f"❌ AMOS imagesTr directory not found: {images_dir}")
        return 1

    nii_files = list(images_dir.glob("amos_*.nii.gz"))
    if not nii_files:
        print(f"❌ No NIfTI files found in {images_dir}")
        return 1

    print(f"✅ Found {len(nii_files)} AMOS NIfTI files")

    # Select test cases
    test_cases = nii_files[:args.max_cases]
    print(f"📋 Testing with {len(test_cases)} cases: {[f.name for f in test_cases]}")

    # Create test data directory
    test_data_dir = output_dir / "test_amos_data"
    test_data_dir.mkdir(exist_ok=True)

    # Copy test cases
    for nii_file in test_cases:
        dest_file = test_data_dir / nii_file.name
        print(f"📋 Copying {nii_file.name} -> {dest_file}")
        # Use symbolic link to save space
        if not dest_file.exists():
            dest_file.symlink_to(nii_file)

    # Create job configuration
    job_config = {
        "job_name": f"comp2comp_inference_test_{args.max_cases}cases",
        "description": f"Test Comp2Comp inference with {args.max_cases} AMOS cases",
        "amos_data_path": str(test_data_dir),
        "output_path": str(output_dir / "results"),
        "max_cases": args.max_cases
    }

    config_file = output_dir / "job_config.json"
    with open(config_file, 'w') as f:
        json.dump(job_config, f, indent=2)

    print(f"📝 Job config saved to: {config_file}")
    print(f"📊 Configuration: {json.dumps(job_config, indent=2)}")

    if args.submit_job:
        print("🚀 Submitting Azure ML job...")

        # Submit job using Azure CLI
        cmd = [
            "az", "ml", "job", "create",
            "--file", ".github/azure-ml/job_step3_comp2comp_inference.yml",
            "--resource-group", "l3-rg",
            "--workspace-name", "l3-vfa-pma-ws",
            "--query", "{name: name, status: status, displayName: display_name}",
            "-o", "json"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            if result.returncode == 0:
                job_info = json.loads(result.stdout)
                print("✅ Job submitted successfully!")
                print(f"📋 Job Name: {job_info['name']}")
                print(f"📋 Display Name: {job_info['displayName']}")
                print(f"📋 Status: {job_info['status']}")
                print(f"🔗 Portal: https://ml.azure.com/runs/{job_info['name']}")

                # Save job info
                job_file = output_dir / "submitted_job.json"
                with open(job_file, 'w') as f:
                    json.dump(job_info, f, indent=2)
                print(f"💾 Job info saved to: {job_file}")

            else:
                print(f"❌ Job submission failed: {result.stderr}")
                return 1

        except Exception as e:
            print(f"❌ Error submitting job: {e}")
            return 1

    else:
        print("💡 To submit the job, run with --submit-job flag")
        print("Example:")
        print(f"    python {sys.argv[0]} --amos-root {args.amos_root} --submit-job")

    print("\n🎯 Next Steps:")
    print("1. Monitor job progress in Azure ML Studio")
    print("2. Download results when job completes")
    print("3. Compare Comp2Comp results with ground truth")
    print("4. Validate VFA/PMA accuracy against radiologist measurements")

    return 0

if __name__ == "__main__":
    sys.exit(main())