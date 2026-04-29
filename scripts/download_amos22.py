#!/usr/bin/env python3
"""
Download AMOS22 data from Azure Storage using Python SDK.
"""
import os
from pathlib import Path
from azure.storage.blob import BlobServiceClient

ACCOUNT_NAME = "l3storage001"
ACCOUNT_KEY = ""
CONTAINER_NAME = "amos22-data"

def main():
    output_dir = Path("data/amos22")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    blob_service_client = BlobServiceClient(
        account_url=f"https://{ACCOUNT_NAME}.blob.core.windows.net",
        credential=ACCOUNT_KEY
    )
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    
    # List all blobs
    blobs = list(container_client.list_blobs())
    print(f"Found {len(blobs)} blobs in container")
    
    for blob in blobs:
        blob_name = blob.name
        local_path = output_dir / blob_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading: {blob_name} ({blob.size / 1024 / 1024:.1f} MB)")
        
        blob_client = container_client.get_blob_client(blob_name)
        with open(local_path, "wb") as f:
            data = blob_client.download_blob()
            f.write(data.readall())
        
        print(f"  -> Saved to {local_path}")
    
    print("Download complete!")

if __name__ == "__main__":
    main()
