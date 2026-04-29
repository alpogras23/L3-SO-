#!/usr/bin/env python3
"""
AMOS22 Data Preparation for Azure Storage
Organizes AMOS22 NIfTI files and TotalSegmentator masks for Azure ML upload
"""

import sys
import shutil
import json
from pathlib import Path
from typing import List, Dict
import argparse

try:
    import SimpleITK as sitk
    import numpy as np
except ImportError:
    print("❌ Import error: SimpleITK not installed")
    print("Install: pip install SimpleITK")
    sys.exit(1)


class AMOSDataPrep:
    """Prepare AMOS22 data for Azure Storage upload"""
    
    def __init__(self, amos_src: Path, ts_masks_src: Path, output_dir: Path):
        self.amos_src = Path(amos_src)
        self.ts_masks_src = Path(ts_masks_src)
        self.output_dir = Path(output_dir)
        
        self.stats = {
            'total_cases': 0,
            'prepared': 0,
            'skipped': 0,
            'errors': []
        }
    
    def find_amos_files(self) -> List[Path]:
        """Find all AMOS22 NIfTI files"""
        patterns = ['*.nii.gz', '*.nii']
        files = []
        for pattern in patterns:
            files.extend(self.amos_src.rglob(pattern))
        
        # Filter out non-AMOS files
        amos_files = [f for f in files if 'amos' in f.stem.lower() or f.stem.isdigit()]
        return amos_files
    
    def find_ts_mask(self, case_id: str) -> Path:
        """Find TotalSegmentator mask for case"""
        patterns = [
            f"{case_id}.nii.gz",
            f"amos_{case_id}.nii.gz",
            f"{case_id}_seg.nii.gz",
            f"segmentations_{case_id}.nii.gz"
        ]
        
        for pattern in patterns:
            files = list(self.ts_masks_src.rglob(pattern))
            if files:
                return files[0]
        
        return None
    
    def validate_file(self, nifti_path: Path) -> Dict:
        """Validate NIfTI file and extract metadata"""
        try:
            img = sitk.ReadImage(str(nifti_path))
            arr = sitk.GetArrayFromImage(img)
            spacing = img.GetSpacing()
            
            return {
                'valid': True,
                'shape': arr.shape,
                'spacing': spacing,
                'dtype': str(arr.dtype),
                'size_mb': nifti_path.stat().st_size / (1024**2)
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def prepare_case(self, amos_file: Path) -> bool:
        """Prepare single case for upload"""
        case_id = amos_file.stem.replace('.nii', '').replace('amos_', '')
        
        print(f"\nProcessing: {case_id}")
        
        # Create case directory
        case_dir = self.output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate AMOS file
        print(f"  Validating AMOS file...")
        amos_info = self.validate_file(amos_file)
        if not amos_info['valid']:
            print(f"  ❌ Invalid AMOS file: {amos_info['error']}")
            self.stats['errors'].append({'case_id': case_id, 'error': amos_info['error']})
            return False
        
        print(f"  ✓ Shape: {amos_info['shape']}, Size: {amos_info['size_mb']:.1f} MB")
        
        # Find TS mask
        ts_mask_file = self.find_ts_mask(case_id)
        if ts_mask_file:
            print(f"  ✓ Found TS mask: {ts_mask_file.name}")
            ts_info = self.validate_file(ts_mask_file)
            if not ts_info['valid']:
                print(f"  ⚠️  TS mask invalid: {ts_info['error']}")
                ts_mask_file = None
        else:
            print(f"  ⚠️  No TS mask found")
        
        # Copy files
        dest_amos = case_dir / f"{case_id}.nii.gz"
        shutil.copy2(amos_file, dest_amos)
        print(f"  ✓ Copied AMOS → {dest_amos.name}")
        
        if ts_mask_file:
            dest_ts = case_dir / f"{case_id}_ts_mask.nii.gz"
            shutil.copy2(ts_mask_file, dest_ts)
            print(f"  ✓ Copied TS mask → {dest_ts.name}")
        
        # Create metadata
        metadata = {
            'case_id': case_id,
            'amos_file': str(dest_amos),
            'ts_mask_file': str(case_dir / f"{case_id}_ts_mask.nii.gz") if ts_mask_file else None,
            'amos_info': amos_info,
            'has_ts_mask': ts_mask_file is not None
        }
        
        with open(case_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Case prepared")
        return True
    
    def run_preparation(self) -> Dict:
        """Run preparation for all cases"""
        print("="*70)
        print("AMOS22 DATA PREPARATION FOR AZURE")
        print("="*70)
        print(f"Source AMOS dir: {self.amos_src}")
        print(f"Source TS masks: {self.ts_masks_src}")
        print(f"Output dir: {self.output_dir}")
        print("="*70)
        
        # Find all AMOS files
        amos_files = self.find_amos_files()
        self.stats['total_cases'] = len(amos_files)
        
        print(f"\nFound {len(amos_files)} AMOS22 cases")
        
        if not amos_files:
            print("❌ No AMOS files found")
            return self.stats
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each case
        for i, amos_file in enumerate(amos_files, 1):
            print(f"\n[{i}/{len(amos_files)}]", end=" ")
            if self.prepare_case(amos_file):
                self.stats['prepared'] += 1
            else:
                self.stats['skipped'] += 1
        
        # Save summary
        self.save_summary()
        
        # Print final statistics
        self.print_summary()
        
        return self.stats
    
    def save_summary(self):
        """Save preparation summary"""
        summary_file = self.output_dir / 'preparation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Create upload instructions
        instructions = f"""
# Azure Storage Upload Instructions

## 1. Install Azure CLI
```bash
brew install azure-cli  # macOS
# or
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash  # Linux
```

## 2. Login to Azure
```bash
az login
```

## 3. Create Storage Account (if not exists)
```bash
az storage account create \\
  --name l3vfapmadata \\
  --resource-group <YOUR_RG> \\
  --location eastus \\
  --sku Standard_LRS
```

## 4. Create Container
```bash
az storage container create \\
  --account-name l3vfapmadata \\
  --name amos22-data \\
  --auth-mode login
```

## 5. Upload Data
```bash
az storage blob upload-batch \\
  --account-name l3vfapmadata \\
  --destination amos22-data \\
  --source {self.output_dir} \\
  --auth-mode login \\
  --overwrite
```

## 6. Verify Upload
```bash
az storage blob list \\
  --account-name l3vfapmadata \\
  --container-name amos22-data \\
  --auth-mode login \\
  --output table
```

## Summary
- Total cases: {self.stats['total_cases']}
- Successfully prepared: {self.stats['prepared']}
- Skipped: {self.stats['skipped']}
- Estimated upload size: ~{self.estimate_size():.1f} GB
"""
        
        with open(self.output_dir / 'UPLOAD_INSTRUCTIONS.md', 'w') as f:
            f.write(instructions)
    
    def estimate_size(self) -> float:
        """Estimate total size in GB"""
        total_size = 0
        for item in self.output_dir.rglob('*.nii.gz'):
            total_size += item.stat().st_size
        return total_size / (1024**3)
    
    def print_summary(self):
        """Print preparation summary"""
        print("\n" + "="*70)
        print("PREPARATION COMPLETE")
        print("="*70)
        print(f"Total cases: {self.stats['total_cases']}")
        print(f"✅ Prepared: {self.stats['prepared']}")
        print(f"⏭️  Skipped: {self.stats['skipped']}")
        print(f"❌ Errors: {len(self.stats['errors'])}")
        print(f"Estimated size: ~{self.estimate_size():.1f} GB")
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Next steps: See {self.output_dir}/UPLOAD_INSTRUCTIONS.md")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Prepare AMOS22 for Azure')
    parser.add_argument('--amos-src', type=str, required=True,
                        help='Source directory with AMOS22 NIfTI files')
    parser.add_argument('--ts-masks', type=str, required=True,
                        help='Source directory with TotalSegmentator masks')
    parser.add_argument('--output-dir', type=str, default='upload_ready',
                        help='Output directory for prepared data')
    
    args = parser.parse_args()
    
    prep = AMOSDataPrep(
        amos_src=Path(args.amos_src),
        ts_masks_src=Path(args.ts_masks),
        output_dir=Path(args.output_dir)
    )
    
    stats = prep.run_preparation()
    
    # Exit with appropriate code
    if stats['prepared'] > 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
