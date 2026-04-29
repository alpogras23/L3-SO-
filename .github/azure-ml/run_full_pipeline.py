#!/usr/bin/env python3
"""
Azure ML Full Pipeline Orchestrator
Runs 3-step pipeline: Teachers → Training → Inference
"""
import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class AMLContext:
    """Azure ML workspace configuration and optional compute override."""

    resource_group: str
    workspace_name: str
    compute_override: str | None = None

def run_command(cmd, desc):
    """Run shell command and print output"""
    print(f"\n{'='*70}")
    print(f"{desc}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {desc} failed!")
        return None
    
    return result.returncode


def build_job_create_cmd(job_file: str, ctx: AMLContext) -> str:
    base_cmd = (
        f"az ml job create --file {job_file} "
        f"--resource-group {ctx.resource_group} "
        f"--workspace-name {ctx.workspace_name}"
    )

    if ctx.compute_override:
        base_cmd += f" --set compute=azureml:{ctx.compute_override}"

    return base_cmd

def submit_job(job_file, ctx: AMLContext, inputs=None):
    """Submit Azure ML job and return job name"""
    cmd = build_job_create_cmd(job_file, ctx)
    
    if inputs:
        for key, value in inputs.items():
            cmd += f' --set inputs.{key}.path="{value}"'
    
    print(f"Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    
    # Parse job name from output
    try:
        output_json = json.loads(result.stdout)
        job_name = output_json.get("name")
        return job_name
    except Exception:
        # Fallback: grep for job name
        import re
        match = re.search(r'"name":\s*"([^"]+)"', result.stdout)
        if match:
            return match.group(1)
    
    return None

def wait_for_job(job_name, ctx: AMLContext, check_interval=60):
    """Wait for job to complete"""
    print(f"\n⏳ Waiting for job: {job_name}")
    print(f"   Check every {check_interval}s")
    print(f"   Portal: https://ml.azure.com/runs/{job_name}")
    
    while True:
        cmd = (
            f"az ml job show --name {job_name} "
            f"--resource-group {ctx.resource_group} "
            f"--workspace-name {ctx.workspace_name} --query status -o tsv"
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            status = result.stdout.strip()
            print(f"   Status: {status}")
            
            if status == "Completed":
                print(f"✅ Job completed: {job_name}\n")
                return True
            elif status == "Failed":
                print(f"❌ Job failed: {job_name}\n")
                return False
            elif status == "Canceled":
                print(f"⚠️  Job canceled: {job_name}\n")
                return False
        
        time.sleep(check_interval)

def get_job_output_path(job_name, output_name, ctx: AMLContext):
    """Get output path from completed job"""
    cmd = (
        f'az ml job show --name {job_name} --resource-group {ctx.resource_group} '
        f'--workspace-name {ctx.workspace_name} --query "outputs.{output_name}.path" -o tsv'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        return result.stdout.strip()
    return None

def main():
    parser = argparse.ArgumentParser(description="Run the L3 VFA/PMA Azure ML pipeline")
    parser.add_argument(
        "--resource-group",
        default="l3-rg",
        help="Azure resource group that contains the ML workspace",
    )
    parser.add_argument(
        "--workspace",
        default="l3-vfa-pma-ws",
        help="Azure ML workspace name",
    )
    parser.add_argument(
        "--compute",
        dest="compute_override",
        default=None,
        help="Optional compute target name that will override YAML definitions",
    )

    args = parser.parse_args()
    ctx = AMLContext(
        resource_group=args.resource_group,
        workspace_name=args.workspace,
        compute_override=args.compute_override,
    )
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║          🚀 L3 VFA/PMA FULL PIPELINE ORCHESTRATOR 🚀                 ║
╚══════════════════════════════════════════════════════════════════════╝

Pipeline Steps:
  1️⃣  TotalSegmentator Teacher Generation (~30 min)
  2️⃣  U-Net Training - 60 epochs (~2-4 hours CPU)
  3️⃣  Production Inference (~10 min)

Total estimated time: ~3-5 hours

""")
    
    # Step 1: Teacher Generation
    print("\n" + "="*70)
    print("STEP 1: TotalSegmentator Teacher Generation")
    print("="*70)
    
    step1_job = submit_job("job_step1_teachers.yml", ctx)
    
    if not step1_job:
        print("❌ Failed to submit Step 1")
        return 1
    
    print(f"✓ Job submitted: {step1_job}")
    
    if not wait_for_job(step1_job, ctx, check_interval=60):
        print("❌ Step 1 failed!")
        return 1
    
    # Get Step 1 output path
    teacher_path = get_job_output_path(step1_job, "teacher_labels", ctx)
    print(f"✓ Teacher labels: {teacher_path}")
    
    # Step 2: Training
    print("\n" + "="*70)
    print("STEP 2: U-Net Training (60 epochs)")
    print("="*70)
    
    step2_job = submit_job("job_step2_training.yml", ctx, inputs={"teacher_labels": teacher_path})
    
    if not step2_job:
        print("❌ Failed to submit Step 2")
        return 1
    
    print(f"✓ Job submitted: {step2_job}")
    
    if not wait_for_job(step2_job, ctx, check_interval=120):
        print("❌ Step 2 failed!")
        return 1
    
    # Get Step 2 output path
    model_path = get_job_output_path(step2_job, "trained_model", ctx)
    print(f"✓ Trained model: {model_path}")
    
    # Step 3: Inference
    print("\n" + "="*70)
    print("STEP 3: Production Inference")
    print("="*70)
    
    step3_job = submit_job("job_step3_inference.yml", ctx, inputs={"trained_model": model_path})
    
    if not step3_job:
        print("❌ Failed to submit Step 3")
        return 1
    
    print(f"✓ Job submitted: {step3_job}")
    
    if not wait_for_job(step3_job, ctx, check_interval=60):
        print("❌ Step 3 failed!")
        return 1
    
    # Get Step 3 output path
    results_path = get_job_output_path(step3_job, "results", ctx)
    print(f"✓ Results: {results_path}")
    
    # Final Summary
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print("="*70)
    print("\nJob IDs:")
    print(f"  Step 1 (Teachers): {step1_job}")
    print(f"  Step 2 (Training): {step2_job}")
    print(f"  Step 3 (Inference): {step3_job}")
    print("\nOutputs:")
    print(f"  Teacher Labels: {teacher_path}")
    print(f"  Trained Model: {model_path}")
    print(f"  Results: {results_path}")
    print("\nDownload results:")
    print(
        "  az ml job download --name "
        f"{step3_job} --resource-group {ctx.resource_group} "
        f"--workspace-name {ctx.workspace_name} --download-path ./results --all"
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
