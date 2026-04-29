#!/usr/bin/env python3
"""
Local Validation Script
Validates VFA/PMA calculations against ground truth (radyologist measurements)
Computes MAE, Pearson correlation, and bias metrics
"""

import sys
import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy as np
    import SimpleITK as sitk
    from scipy import stats
    from core_mini import (
        load_hu, find_l3_index, get_vertebra_center,
        inner_abdomen_via_wall, compute_vfa_pma_from_hu
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install: pip install numpy scipy SimpleITK")
    sys.exit(1)


class LocalValidator:
    """Validates processing against ground truth measurements"""
    
    def __init__(self, amos_dir: Path, gt_csv: Path, output_dir: Path):
        self.amos_dir = Path(amos_dir)
        self.gt_csv = Path(gt_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ground_truth = self.load_ground_truth()
        self.results = []
        
    def load_ground_truth(self) -> Dict[str, Dict]:
        """Load ground truth CSV file"""
        gt = {}
        
        if not self.gt_csv.exists():
            print(f"⚠️  Ground truth file not found: {self.gt_csv}")
            return gt
        
        with open(self.gt_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                case_id = row.get('case_id') or row.get('id') or row.get('patient_id')
                if case_id:
                    gt[case_id] = {
                        'VFA_cm2': float(row.get('VFA_cm2', 0) or row.get('VFA', 0)),
                        'PMA_cm2': float(row.get('PMA_cm2', 0) or row.get('PMA', 0)),
                        'height_m': float(row.get('height_m', 1.7) or row.get('height', 1.7)),
                        'sex': row.get('sex', 'M') or row.get('gender', 'M')
                    }
        
        print(f"✓ Loaded ground truth for {len(gt)} cases")
        return gt
    
    def find_case_file(self, case_id: str) -> Optional[Path]:
        """Find NIfTI file for case"""
        # Try common patterns
        patterns = [
            f"{case_id}.nii.gz",
            f"{case_id}.nii",
            f"amos_{case_id}.nii.gz",
            f"case_{case_id}.nii.gz",
        ]
        
        for pattern in patterns:
            files = list(self.amos_dir.rglob(pattern))
            if files:
                return files[0]
        
        return None
    
    def process_case(self, case_id: str, gt_data: Dict) -> Optional[Dict]:
        """Process single case and compare with ground truth"""
        print(f"\n{'='*70}")
        print(f"Validating: {case_id}")
        print(f"{'='*70}")
        
        # Find file
        case_file = self.find_case_file(case_id)
        if case_file is None:
            print(f"❌ File not found for {case_id}")
            return None
        
        print(f"File: {case_file.name}")
        
        try:
            # Load image
            img = sitk.ReadImage(str(case_file))
            hu_array = sitk.GetArrayFromImage(img)
            spacing = img.GetSpacing()
            spacing = (spacing[2], spacing[0], spacing[1])
            
            # Find L3
            l3_idx = find_l3_index(hu_array, spacing)
            if l3_idx is None:
                print(f"❌ L3 detection failed")
                return None
            
            # Get vertebra center
            vb_center, vb_conf = get_vertebra_center(
                hu_array[l3_idx], 
                spacing[1:],
                return_confidence=True
            )
            
            if vb_center is None or vb_conf < 0.5:
                print(f"❌ Vertebra center failed (conf={vb_conf})")
                return None
            
            # Segment inner abdomen
            inner_mask = inner_abdomen_via_wall(
                hu_array[l3_idx],
                vb_center,
                spacing[1:]
            )
            
            if inner_mask is None:
                print(f"❌ Fascia segmentation failed")
                return None
            
            # Compute VFA/PMA
            metrics = compute_vfa_pma_from_hu(
                hu_array[l3_idx],
                inner_mask,
                vb_center,
                spacing[1:],
                height_m=gt_data['height_m'],
                sex=gt_data['sex']
            )
            
            if metrics is None:
                print(f"❌ VFA/PMA computation failed")
                return None
            
            # Compare with ground truth
            pred_vfa = metrics.get('VFA_cm2', 0)
            pred_pma = metrics.get('PMA_cm2', 0)
            gt_vfa = gt_data['VFA_cm2']
            gt_pma = gt_data['PMA_cm2']
            
            vfa_error = abs(pred_vfa - gt_vfa)
            pma_error = abs(pred_pma - gt_pma)
            vfa_bias = pred_vfa - gt_vfa
            pma_bias = pred_pma - gt_pma
            
            result = {
                'case_id': case_id,
                'predicted_VFA_cm2': float(pred_vfa),
                'predicted_PMA_cm2': float(pred_pma),
                'gt_VFA_cm2': float(gt_vfa),
                'gt_PMA_cm2': float(gt_pma),
                'VFA_error': float(vfa_error),
                'PMA_error': float(pma_error),
                'VFA_bias': float(vfa_bias),
                'PMA_bias': float(pma_bias),
                'L3_index': int(l3_idx),
                'VB_confidence': float(vb_conf),
                'status': 'success'
            }
            
            print(f"✓ Predicted VFA: {pred_vfa:.1f} cm² (GT: {gt_vfa:.1f}, Error: {vfa_error:.1f})")
            print(f"✓ Predicted PMA: {pred_pma:.1f} cm² (GT: {gt_pma:.1f}, Error: {pma_error:.1f})")
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_validation(self) -> Dict:
        """Run validation on all ground truth cases"""
        print(f"\n{'='*70}")
        print("STARTING LOCAL VALIDATION")
        print(f"AMOS directory: {self.amos_dir}")
        print(f"Ground truth: {self.gt_csv}")
        print(f"Cases to validate: {len(self.ground_truth)}")
        print(f"{'='*70}\n")
        
        for case_id, gt_data in self.ground_truth.items():
            result = self.process_case(case_id, gt_data)
            if result:
                self.results.append(result)
        
        # Compute summary metrics
        summary = self.compute_metrics()
        
        # Save results
        self.save_results(summary)
        
        # Print summary
        self.print_summary(summary)
        
        return summary
    
    def compute_metrics(self) -> Dict:
        """Compute validation metrics"""
        if not self.results:
            return {
                'total_cases': 0,
                'successful': 0,
                'metrics': {}
            }
        
        # Extract arrays
        pred_vfa = np.array([r['predicted_VFA_cm2'] for r in self.results])
        gt_vfa = np.array([r['gt_VFA_cm2'] for r in self.results])
        pred_pma = np.array([r['predicted_PMA_cm2'] for r in self.results])
        gt_pma = np.array([r['gt_PMA_cm2'] for r in self.results])
        
        # VFA metrics
        vfa_mae = np.mean(np.abs(pred_vfa - gt_vfa))
        vfa_rmse = np.sqrt(np.mean((pred_vfa - gt_vfa)**2))
        vfa_bias = np.mean(pred_vfa - gt_vfa)
        vfa_pearson_r, vfa_p_value = stats.pearsonr(pred_vfa, gt_vfa)
        
        # PMA metrics
        pma_mae = np.mean(np.abs(pred_pma - gt_pma))
        pma_rmse = np.sqrt(np.mean((pred_pma - gt_pma)**2))
        pma_bias = np.mean(pred_pma - gt_pma)
        pma_pearson_r, pma_p_value = stats.pearsonr(pred_pma, gt_pma)
        
        summary = {
            'total_cases': len(self.ground_truth),
            'successful': len(self.results),
            'success_rate': len(self.results) / len(self.ground_truth) * 100,
            'metrics': {
                'VFA': {
                    'MAE': float(vfa_mae),
                    'RMSE': float(vfa_rmse),
                    'bias': float(vfa_bias),
                    'pearson_r': float(vfa_pearson_r),
                    'p_value': float(vfa_p_value)
                },
                'PMA': {
                    'MAE': float(pma_mae),
                    'RMSE': float(pma_rmse),
                    'bias': float(pma_bias),
                    'pearson_r': float(pma_pearson_r),
                    'p_value': float(pma_p_value)
                }
            },
            'results': self.results
        }
        
        return summary
    
    def save_results(self, summary: Dict):
        """Save results to files"""
        # Save JSON summary
        summary_file = self.output_dir / 'validation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save CSV with detailed results
        csv_file = self.output_dir / 'validation_results.csv'
        with open(csv_file, 'w', newline='') as f:
            fieldnames = ['case_id', 'predicted_VFA_cm2', 'gt_VFA_cm2', 'VFA_error', 
                          'predicted_PMA_cm2', 'gt_PMA_cm2', 'PMA_error', 
                          'VB_confidence', 'status']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow({k: r[k] for k in fieldnames})
        
        print(f"\n✓ Results saved:")
        print(f"  {summary_file}")
        print(f"  {csv_file}")
    
    def print_summary(self, summary: Dict):
        """Print validation summary"""
        print(f"\n{'='*70}")
        print("VALIDATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total cases: {summary['total_cases']}")
        print(f"✅ Successful: {summary['successful']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print(f"\nVFA Metrics:")
        print(f"  MAE:        {summary['metrics']['VFA']['MAE']:.2f} cm²")
        print(f"  RMSE:       {summary['metrics']['VFA']['RMSE']:.2f} cm²")
        print(f"  Bias:       {summary['metrics']['VFA']['bias']:.2f} cm²")
        print(f"  Pearson r:  {summary['metrics']['VFA']['pearson_r']:.3f}")
        print(f"\nPMA Metrics:")
        print(f"  MAE:        {summary['metrics']['PMA']['MAE']:.2f} cm²")
        print(f"  RMSE:       {summary['metrics']['PMA']['RMSE']:.2f} cm²")
        print(f"  Bias:       {summary['metrics']['PMA']['bias']:.2f} cm²")
        print(f"  Pearson r:  {summary['metrics']['PMA']['pearson_r']:.3f}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Local Validation')
    parser.add_argument('--amos-dir', type=str, required=True,
                        help='Directory containing AMOS22 NIfTI files')
    parser.add_argument('--gt', type=str, required=True,
                        help='Ground truth CSV file (case_id, VFA_cm2, PMA_cm2, height_m, sex)')
    parser.add_argument('--output-dir', type=str, default='validation_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    validator = LocalValidator(
        amos_dir=Path(args.amos_dir),
        gt_csv=Path(args.gt),
        output_dir=Path(args.output_dir)
    )
    
    summary = validator.run_validation()
    
    # Exit with success if validation metrics are acceptable
    vfa_mae = summary['metrics']['VFA']['MAE']
    pma_mae = summary['metrics']['PMA']['MAE']
    
    if vfa_mae < 50 and pma_mae < 15:  # Acceptable thresholds
        print("✅ Validation PASSED (MAE within acceptable range)")
        sys.exit(0)
    else:
        print("⚠️  Validation WARNING (MAE exceeds acceptable range)")
        sys.exit(1)


if __name__ == '__main__':
    main()
