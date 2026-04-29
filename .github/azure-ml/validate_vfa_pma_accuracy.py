#!/usr/bin/env python3
"""
VFA/PMA Validation Script
Compares inference results with radiologist ground truth measurements.

Usage:
    python validate_vfa_pma_accuracy.py \
        --inference-results ./inference_results \
        --ground-truth ground_truth.csv \
        --output validation_report.json
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_inference_results(results_dir: str) -> pd.DataFrame:
    """Load all inference results from JSON files."""
    logger.info(f"Loading inference results from {results_dir}")
    
    results = []
    results_path = Path(results_dir)
    
    for json_file in sorted(results_path.glob("*_vfa_pma.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
                if 'case_id' in data and 'pma_mm2' in data:
                    results.append({
                        'case_id': data['case_id'],
                        'pma_inferred': data['pma_mm2'],
                        'vat_inferred': data['vat_mm2'],
                        'sat_inferred': data['sat_mm2'],
                    })
        except Exception as e:
            logger.warning(f"Error reading {json_file}: {e}")
    
    df = pd.DataFrame(results)
    logger.info(f"Loaded {len(df)} cases")
    return df


def load_ground_truth(gt_file: str) -> pd.DataFrame:
    """Load ground truth from CSV."""
    logger.info(f"Loading ground truth from {gt_file}")
    
    df = pd.read_csv(gt_file)
    
    # Standardize column names
    rename_map = {
        'case_id': 'case_id',
        'case': 'case_id',
        'pma': 'pma_gt',
        'pma_mm2': 'pma_gt',
        'vat': 'vat_gt',
        'vat_mm2': 'vat_gt',
        'sat': 'sat_gt',
        'sat_mm2': 'sat_gt',
    }
    
    df.rename(columns=rename_map, inplace=True)
    
    # Ensure case_id is string
    if 'case_id' in df.columns:
        df['case_id'] = df['case_id'].astype(str)
    
    logger.info(f"Loaded ground truth for {len(df)} cases")
    return df


def merge_results(inferred_df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    """Merge inference results with ground truth."""
    merged = pd.merge(inferred_df, gt_df, on='case_id', how='inner')
    
    logger.info(f"Matched {len(merged)} cases with ground truth")
    
    if len(merged) < len(inferred_df):
        missing = len(inferred_df) - len(merged)
        logger.warning(f"⚠️ {missing} inferred cases not in ground truth")
    
    return merged


def calculate_metrics(merged_df: pd.DataFrame) -> Dict:
    """Calculate accuracy metrics."""
    metrics = {}
    
    for tissue in ['pma', 'vat', 'sat']:
        col_inferred = f'{tissue}_inferred'
        col_gt = f'{tissue}_gt'
        
        if col_inferred not in merged_df.columns or col_gt not in merged_df.columns:
            continue
        
        inferred = merged_df[col_inferred]
        gt = merged_df[col_gt]
        
        # Remove NaN values
        valid_mask = ~(inferred.isna() | gt.isna())
        inferred = inferred[valid_mask]
        gt = gt[valid_mask]
        
        if len(inferred) == 0:
            logger.warning(f"No valid data for {tissue}")
            continue
        
        # Calculate metrics
        mae = mean_absolute_error(gt, inferred)
        rmse = np.sqrt(mean_squared_error(gt, inferred))
        
        # Correlation
        r_pearson, p_pearson = pearsonr(inferred, gt)
        r_spearman, p_spearman = spearmanr(inferred, gt)
        
        # Bias
        bias = (inferred - gt).mean()
        bias_pct = (bias / gt.mean()) * 100
        
        # Limits of agreement (Bland-Altman)
        diff = inferred - gt
        mean_diff = diff.mean()
        std_diff = diff.std()
        upper_loa = mean_diff + 1.96 * std_diff
        lower_loa = mean_diff - 1.96 * std_diff
        
        metrics[tissue] = {
            'n': len(inferred),
            'mae': float(mae),
            'rmse': float(rmse),
            'bias': float(bias),
            'bias_pct': float(bias_pct),
            'r_pearson': float(r_pearson),
            'p_pearson': float(p_pearson),
            'r_spearman': float(r_spearman),
            'p_spearman': float(p_spearman),
            'loa_upper': float(upper_loa),
            'loa_lower': float(lower_loa),
            'loa_mean': float(mean_diff),
        }
    
    return metrics


def assess_accuracy(metrics: Dict) -> Dict[str, str]:
    """Assess accuracy against clinical standards."""
    assessment = {}
    
    # Standard thresholds
    THRESHOLDS = {
        'pma': {'mae': 50, 'bias_pct': 10, 'r': 0.90},
        'vat': {'mae': 100, 'bias_pct': 10, 'r': 0.90},
        'sat': {'mae': 150, 'bias_pct': 15, 'r': 0.85},
    }
    
    for tissue, tissue_metrics in metrics.items():
        threshold = THRESHOLDS.get(tissue, {})
        
        mae_ok = tissue_metrics['mae'] <= threshold.get('mae', 1e9)
        bias_ok = abs(tissue_metrics['bias_pct']) <= threshold.get('bias_pct', 100)
        r_ok = tissue_metrics['r_pearson'] >= threshold.get('r', 0)
        
        if mae_ok and bias_ok and r_ok:
            assessment[tissue] = "✅ PASS"
        elif r_ok and bias_ok:
            assessment[tissue] = "⚠️ BORDERLINE (slightly high MAE)"
        else:
            assessment[tissue] = "❌ FAIL"
    
    return assessment


def generate_report(merged_df: pd.DataFrame, metrics: Dict, assessment: Dict) -> Dict:
    """Generate comprehensive validation report."""
    
    report = {
        'summary': {
            'total_cases': len(merged_df),
            'timestamp': pd.Timestamp.now().isoformat(),
        },
        'metrics': metrics,
        'assessment': assessment,
        'recommendations': [],
    }
    
    # Add recommendations
    for tissue, status in assessment.items():
        if '❌' in status:
            if tissue == 'pma':
                report['recommendations'].append(
                    "PMA accuracy below threshold. "
                    "Consider adjusting muscle HU thresholds (-29 to +150 HU) "
                    "or retraining the U-Net model."
                )
            elif tissue == 'vat':
                report['recommendations'].append(
                    "VAT accuracy below threshold. "
                    "Consider adjusting visceral fat HU thresholds (-150 to -50 HU) "
                    "or refining peritoneal boundary detection."
                )
        elif '⚠️' in status:
            report['recommendations'].append(
                f"{tissue.upper()} accuracy is acceptable but near threshold. "
                f"Monitor for systematic bias in future applications."
            )
    
    if not report['recommendations']:
        report['recommendations'].append(
            "✅ All tissue measurements meet clinical accuracy standards. "
            "Pipeline is ready for production use."
        )
    
    return report


def plot_bland_altman(merged_df: pd.DataFrame, metrics: Dict, output_dir: str):
    """Generate Bland-Altman plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    tissues = ['pma', 'vat', 'sat']
    
    for ax, tissue in zip(axes, tissues):
        col_inferred = f'{tissue}_inferred'
        col_gt = f'{tissue}_gt'
        
        if col_inferred not in merged_df.columns:
            continue
        
        inferred = merged_df[col_inferred]
        gt = merged_df[col_gt]
        
        # Bland-Altman plot
        mean = (inferred + gt) / 2
        diff = inferred - gt
        
        ax.scatter(mean, diff, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--', label='Perfect agreement')
        ax.axhline(y=metrics[tissue]['loa_mean'], color='b', linestyle='--', 
                  label='Mean difference')
        ax.axhline(y=metrics[tissue]['loa_upper'], color='g', linestyle=':', 
                  label='±1.96 SD')
        ax.axhline(y=metrics[tissue]['loa_lower'], color='g', linestyle=':')
        
        ax.set_xlabel(f'{tissue.upper()} Mean (mm²)')
        ax.set_ylabel('Difference (mm²)')
        ax.set_title(f'{tissue.upper()} - Bland-Altman Plot')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = output_path / 'bland_altman.png'
    plt.savefig(output_file, dpi=150)
    logger.info(f"Saved Bland-Altman plot: {output_file}")
    plt.close()


def main():
    """Main validation pipeline."""
    parser = argparse.ArgumentParser(description="VFA/PMA Accuracy Validation")
    parser.add_argument('--inference-results', type=str, required=True,
                       help='Directory with inference results')
    parser.add_argument('--ground-truth', type=str, required=True,
                       help='CSV file with ground truth measurements')
    parser.add_argument('--output', type=str, default='validation_report.json',
                       help='Output report file')
    parser.add_argument('--plot-dir', type=str, default='./plots',
                       help='Directory for plots')
    
    args = parser.parse_args()
    
    # Load data
    inferred_df = load_inference_results(args.inference_results)
    gt_df = load_ground_truth(args.ground_truth)
    
    # Merge
    merged_df = merge_results(inferred_df, gt_df)
    
    # Calculate metrics
    metrics = calculate_metrics(merged_df)
    
    # Assess
    assessment = assess_accuracy(metrics)
    
    # Generate report
    report = generate_report(merged_df, metrics, assessment)
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {args.output}")
    
    # Generate plots
    plot_bland_altman(merged_df, metrics, args.plot_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("VFA/PMA VALIDATION REPORT")
    print("="*70)
    
    for tissue in ['pma', 'vat', 'sat']:
        if tissue not in metrics:
            continue
        
        m = metrics[tissue]
        status = assessment[tissue]
        
        print(f"\n{tissue.upper():>3} ({status})")
        print(f"  MAE:          {m['mae']:>8.1f} mm²")
        print(f"  Bias:         {m['bias']:>+8.1f} mm² ({m['bias_pct']:+.1f}%)")
        print(f"  r (Pearson):  {m['r_pearson']:>8.3f} (p={m['p_pearson']:.3e})")
        print(f"  LoA:          [{m['loa_lower']:>8.1f}, {m['loa_upper']:>8.1f}]")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
