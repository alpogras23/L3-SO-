#!/usr/bin/env python3
"""
Azure ML Job Monitor
Monitors job status and downloads results when completed
"""
import time
import json
import subprocess
import sys
from pathlib import Path

JOB_NAME = "sharp_lobster_45mgy3zkb7"
RG = "l3-rg"
WS = "l3-vfa-pma-ws"
OUTPUT_DIR = Path("../../job_outputs")

def get_job_status():
    """Get current job status"""
    cmd = [
        "az", "ml", "job", "show",
        "--name", JOB_NAME,
        "--resource-group", RG,
        "--workspace-name", WS,
        "--query", "{Name:name, Status:status, StartTime:creation_context.created_at}"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return json.loads(result.stdout)
    return None

def download_outputs():
    """Download job outputs"""
    output_path = OUTPUT_DIR / JOB_NAME
    output_path.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "az", "ml", "job", "download",
        "--name", JOB_NAME,
        "--resource-group", RG,
        "--workspace-name", WS,
        "--download-path", str(output_path),
        "--all"
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0

def main():
    print("=" * 70)
    print(f"Monitoring Azure ML Job: {JOB_NAME}")
    print("=" * 70)
    print(f"\n🔗 Portal URL:")
    print(f"https://ml.azure.com/runs/{JOB_NAME}?wsid=/subscriptions/d2d593d8-01af-4128-98e5-31ede386359f/resourcegroups/{RG}/workspaces/{WS}\n")
    
    last_status = None
    check_interval = 30  # seconds
    
    while True:
        status_info = get_job_status()
        
        if status_info is None:
            print("❌ Error getting job status")
            time.sleep(check_interval)
            continue
        
        current_status = status_info.get("Status")
        
        if current_status != last_status:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] Status: {current_status}")
            last_status = current_status
        
        # Terminal states
        if current_status == "Completed":
            print("\n✅ Job completed successfully!")
            print("\n📥 Downloading outputs...")
            if download_outputs():
                print(f"✅ Outputs downloaded to: {OUTPUT_DIR / JOB_NAME}")
            else:
                print("❌ Error downloading outputs")
            break
        
        elif current_status == "Failed":
            print("\n❌ Job failed!")
            print("\n📥 Downloading logs...")
            download_outputs()
            break
        
        elif current_status == "Canceled":
            print("\n⚠️ Job was canceled")
            break
        
        # Wait before next check
        time.sleep(check_interval)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Monitoring interrupted. Job is still running in Azure.")
        print(f"Check status: az ml job show --name {JOB_NAME} --resource-group {RG} --workspace-name {WS}")
        sys.exit(0)
