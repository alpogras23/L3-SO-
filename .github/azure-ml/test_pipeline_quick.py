#!/usr/bin/env python3
"""
Quick Pipeline Test Script
Tests VFA/PMA calculation on 5-10 cases locally using core_mini.py
No Azure ML required - runs on desktop for fast validation
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy as np
    import SimpleITK as sitk
    # Import from core_mini (stable production core)
    from core_mini import (
        load_hu, find_l3_index, get_vertebra_center,
        inner_abdomen_via_wall, compute_vfa_pma_from_hu
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the correct environment with dependencies installed")
    sys.exit(1)


class QuickPipelineTest:
    """Quick test runner for L3 VFA/PMA pipeline"""
    
    def __init__(self, test_cases_dir: Path, output_dir: Path, max_cases: int = 5):
        self.test_cases_dir = Path(test_cases_dir)
        self.output_dir = Path(output_dir)
        self.max_cases = max_cases
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        self.errors = []
        
    def find_test_cases(self) -> List[Path]:
        """Find DICOM or NIfTI files in test directory"""
        dicom_files = list(self.test_cases_dir.rglob("*.dcm"))
        nifti_files = list(self.test_cases_dir.rglob("*.nii.gz"))
        nifti_files += list(self.test_cases_dir.rglob("*.nii"))
        
        all_files = dicom_files + nifti_files
        
        if not all_files:
            print(f"⚠️  No DICOM or NIfTI files found in {self.test_cases_dir}")
            return []
        
        # Limit to max_cases
        selected = all_files[:self.max_cases]
        print(f"✓ Found {len(all_files)} files, testing first {len(selected)}")
        return selected
    
    def load_image(self, file_path: Path) -> Tuple[Optional[np.ndarray], Optional[Tuple]]:
        """Load HU array and spacing from DICOM or NIfTI"""
        try:
            if file_path.suffix == '.dcm':
                # Single DICOM file
                hu_array, spacing = load_hu(str(file_path))
            else:
                # NIfTI file
                img = sitk.ReadImage(str(file_path))
                hu_array = sitk.GetArrayFromImage(img)
                spacing = img.GetSpacing()  # (x, y, z)
                spacing = (spacing[2], spacing[0], spacing[1])  # Convert to (z, y, x)
            
            return hu_array, spacing
        except Exception as e:
            print(f"❌ Load error for {file_path.name}: {e}")
            return None, None
    
    def process_case(self, file_path: Path, height_m: float = 1.7, sex: str = 'M') -> Dict:
        """Process single case through full pipeline"""
        case_id = file_path.stem
        print(f"\n{'='*70}")
        print(f"Processing: {case_id}")
        print(f"{'='*70}")
        
        start_time = time.time()
        result = {
            'case_id': case_id,
            'file_path': str(file_path),
            'status': 'unknown',
            'error': None,
            'processing_time_sec': 0.0
        }
        
        try:
            # Step 1: Load image
            print("Step 1/5: Loading image...")
            hu_array, spacing = self.load_image(file_path)
            if hu_array is None:
                result['status'] = 'failed'
                result['error'] = 'Failed to load image'
                return result
            
            print(f"  ✓ Loaded: shape={hu_array.shape}, spacing={spacing}")
            
            # Step 2: Find L3
            print("Step 2/5: Finding L3 slice...")
            l3_idx = find_l3_index(hu_array, spacing)
            if l3_idx is None:
                result['status'] = 'failed'
                result['error'] = 'L3 detection failed'
                return result
            
            print(f"  ✓ L3 index: {l3_idx}")
            result['L3_index'] = int(l3_idx)
            
            # Step 3: Get vertebra center
            print("Step 3/5: Finding vertebra center...")
            vb_center, vb_conf = get_vertebra_center(
                hu_array[l3_idx], 
                spacing[1:],  # (dy, dx)
                return_confidence=True
            )
            
            if vb_center is None:
                result['status'] = 'failed'
                result['error'] = 'Vertebra center detection failed'
                return result
            
            print(f"  ✓ Vertebra center: {vb_center}, confidence: {vb_conf:.3f}")
            result['VB_center'] = [float(vb_center[0]), float(vb_center[1])]
            result['VB_confidence'] = float(vb_conf)
            
            # Step 4: Inner abdomen segmentation
            print("Step 4/5: Segmenting inner abdomen...")
            inner_mask = inner_abdomen_via_wall(
                hu_array[l3_idx],
                vb_center,
                spacing[1:]
            )
            
            if inner_mask is None:
                result['status'] = 'failed'
                result['error'] = 'Fascia segmentation failed'
                return result
            
            print(f"  ✓ Inner abdomen area: {np.sum(inner_mask)} pixels")
            
            # Step 5: Compute VFA/PMA
            print("Step 5/5: Computing VFA/PMA...")
            metrics = compute_vfa_pma_from_hu(
                hu_array[l3_idx],
                inner_mask,
                vb_center,
                spacing[1:],
                height_m=height_m,
                sex=sex
            )
            
            if metrics is None:
                result['status'] = 'failed'
                result['error'] = 'VFA/PMA computation failed'
                return result
            
            # Extract metrics
            result['VFA_mm2'] = float(metrics.get('VFA_mm2', 0))
            result['VFA_cm2'] = float(metrics.get('VFA_cm2', 0))
            result['PMA_mm2'] = float(metrics.get('PMA_mm2', 0))
            result['PMA_cm2'] = float(metrics.get('PMA_cm2', 0))
            result['VFA_PMA_ratio'] = float(metrics.get('VFA_PMA_ratio', 0))
            result['status'] = 'success'
            
            print(f"\n✅ SUCCESS:")
            print(f"  VFA: {result['VFA_cm2']:.1f} cm²")
            print(f"  PMA: {result['PMA_cm2']:.1f} cm²")
            print(f"  Ratio: {result['VFA_PMA_ratio']:.2f}")
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            result['processing_time_sec'] = time.time() - start_time
            print(f"⏱️  Processing time: {result['processing_time_sec']:.2f}s")
        
        return result
    
    def run_tests(self, height_m: float = 1.7, sex: str = 'M') -> Dict:
        """Run tests on all cases"""
        test_cases = self.find_test_cases()
        
        if not test_cases:
            return {
                'status': 'no_cases_found',
                'total_cases': 0,
                'successful': 0,
                'failed': 0,
                'results': []
            }
        
        print(f"\n{'='*70}")
        print(f"STARTING QUICK PIPELINE TEST")
        print(f"Test cases: {len(test_cases)}")
        print(f"Output dir: {self.output_dir}")
        print(f"{'='*70}\n")
        
        for i, case_file in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] Testing: {case_file.name}")
            result = self.process_case(case_file, height_m, sex)
            self.results.append(result)
            
            # Save individual result
            result_file = self.output_dir / f"{result['case_id']}_result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
        
        # Generate summary
        summary = self.generate_summary()
        
        # Save summary
        summary_file = self.output_dir / "test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"TEST COMPLETE")
        print(f"{'='*70}")
        print(f"Total: {summary['total_cases']}")
        print(f"✅ Success: {summary['successful']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print(f"\nResults saved to: {self.output_dir}")
        print(f"{'='*70}\n")
        
        return summary
    
    def generate_summary(self) -> Dict:
        """Generate summary statistics"""
        total = len(self.results)
        successful = sum(1 for r in self.results if r['status'] == 'success')
        failed = total - successful
        
        # Compute statistics for successful cases
        vfa_values = [r['VFA_cm2'] for r in self.results if r['status'] == 'success']
        pma_values = [r['PMA_cm2'] for r in self.results if r['status'] == 'success']
        
        summary = {
            'total_cases': total,
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / total * 100) if total > 0 else 0,
            'statistics': {}
        }
        
        if vfa_values:
            summary['statistics']['VFA_cm2'] = {
                'mean': float(np.mean(vfa_values)),
                'std': float(np.std(vfa_values)),
                'min': float(np.min(vfa_values)),
                'max': float(np.max(vfa_values))
            }
        
        if pma_values:
            summary['statistics']['PMA_cm2'] = {
                'mean': float(np.mean(pma_values)),
                'std': float(np.std(pma_values)),
                'min': float(np.min(pma_values)),
                'max': float(np.max(pma_values))
            }
        
        summary['results'] = self.results
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Quick Pipeline Test')
    parser.add_argument('--cases-dir', type=str, required=True,
                        help='Directory containing test DICOM/NIfTI files')
    parser.add_argument('--output-dir', type=str, default='test_results',
                        help='Output directory for results')
    parser.add_argument('--max-cases', type=int, default=5,
                        help='Maximum number of cases to test')
    parser.add_argument('--height', type=float, default=1.7,
                        help='Patient height in meters (default: 1.7)')
    parser.add_argument('--sex', type=str, default='M', choices=['M', 'F'],
                        help='Patient sex (M or F)')
    
    args = parser.parse_args()
    
    # Run tests
    tester = QuickPipelineTest(
        test_cases_dir=args.cases_dir,
        output_dir=Path(args.output_dir),
        max_cases=args.max_cases
    )
    
    summary = tester.run_tests(height_m=args.height, sex=args.sex)
    
    # Exit with appropriate code
    if summary['failed'] == 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
