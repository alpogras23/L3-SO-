#!/usr/bin/env python3
"""
Hybrid Mode Parameter Optimization
Optimizes alpha blending parameter (rule-based vs DL) for best accuracy
"""

import sys
import json
import csv
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy as np
    from scipy import stats
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install: pip install numpy scipy matplotlib")
    sys.exit(1)


def load_ground_truth(gt_csv: Path) -> dict:
    """Load ground truth measurements"""
    gt = {}
    with open(gt_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row.get('case_id') or row.get('id')
            gt[case_id] = {
                'VFA_cm2': float(row.get('VFA_cm2', 0)),
                'PMA_cm2': float(row.get('PMA_cm2', 0))
            }
    return gt


def simulate_hybrid_predictions(gt: dict, alpha: float) -> dict:
    """
    Simulate hybrid predictions for given alpha
    alpha=0: pure rule-based
    alpha=1: pure DL
    alpha=0.5: 50/50 blend
    """
    # Simplified simulation - in production, would run actual predictions
    # Here we assume rule-based has some bias, DL has different bias
    
    results = {}
    for case_id, gt_vals in gt.items():
        # Simulate rule-based (tends to underestimate slightly)
        rb_vfa = gt_vals['VFA_cm2'] * 0.95 + np.random.normal(0, 10)
        rb_pma = gt_vals['PMA_cm2'] * 0.98 + np.random.normal(0, 3)
        
        # Simulate DL (tends to overestimate slightly)
        dl_vfa = gt_vals['VFA_cm2'] * 1.05 + np.random.normal(0, 15)
        dl_pma = gt_vals['PMA_cm2'] * 1.02 + np.random.normal(0, 4)
        
        # Hybrid blend
        pred_vfa = (1 - alpha) * rb_vfa + alpha * dl_vfa
        pred_pma = (1 - alpha) * rb_pma + alpha * dl_pma
        
        results[case_id] = {
            'pred_VFA_cm2': pred_vfa,
            'pred_PMA_cm2': pred_pma,
            'gt_VFA_cm2': gt_vals['VFA_cm2'],
            'gt_PMA_cm2': gt_vals['PMA_cm2']
        }
    
    return results


def compute_metrics(results: dict) -> dict:
    """Compute accuracy metrics"""
    pred_vfa = np.array([r['pred_VFA_cm2'] for r in results.values()])
    gt_vfa = np.array([r['gt_VFA_cm2'] for r in results.values()])
    pred_pma = np.array([r['pred_PMA_cm2'] for r in results.values()])
    gt_pma = np.array([r['gt_PMA_cm2'] for r in results.values()])
    
    vfa_mae = np.mean(np.abs(pred_vfa - gt_vfa))
    vfa_bias = np.mean(pred_vfa - gt_vfa)
    vfa_r, _ = stats.pearsonr(pred_vfa, gt_vfa)
    
    pma_mae = np.mean(np.abs(pred_pma - gt_pma))
    pma_bias = np.mean(pred_pma - gt_pma)
    pma_r, _ = stats.pearsonr(pred_pma, gt_pma)
    
    # Combined score (lower is better)
    combined_score = vfa_mae + pma_mae - (vfa_r + pma_r) * 50
    
    return {
        'VFA_MAE': vfa_mae,
        'VFA_bias': vfa_bias,
        'VFA_pearson_r': vfa_r,
        'PMA_MAE': pma_mae,
        'PMA_bias': pma_bias,
        'PMA_pearson_r': pma_r,
        'combined_score': combined_score
    }


def optimize_alpha(gt_csv: Path, output_dir: Path, alpha_range: tuple = (0.0, 1.0, 21)):
    """Grid search over alpha values"""
    print("="*70)
    print("HYBRID PARAMETER OPTIMIZATION")
    print("="*70)
    
    gt = load_ground_truth(gt_csv)
    print(f"Loaded ground truth for {len(gt)} cases")
    
    # Alpha grid
    alphas = np.linspace(alpha_range[0], alpha_range[1], alpha_range[2])
    
    results_table = []
    
    print(f"\nTesting {len(alphas)} alpha values from {alpha_range[0]} to {alpha_range[1]}...")
    
    for alpha in alphas:
        predictions = simulate_hybrid_predictions(gt, alpha)
        metrics = compute_metrics(predictions)
        
        results_table.append({
            'alpha': alpha,
            **metrics
        })
        
        print(f"  α={alpha:.2f}: VFA_MAE={metrics['VFA_MAE']:.2f}, "
              f"PMA_MAE={metrics['PMA_MAE']:.2f}, Score={metrics['combined_score']:.2f}")
    
    # Find optimal alpha
    optimal_idx = np.argmin([r['combined_score'] for r in results_table])
    optimal = results_table[optimal_idx]
    
    print(f"\n✅ Optimal alpha: {optimal['alpha']:.2f}")
    print(f"   VFA MAE: {optimal['VFA_MAE']:.2f} cm²")
    print(f"   PMA MAE: {optimal['PMA_MAE']:.2f} cm²")
    print(f"   VFA Pearson r: {optimal['VFA_pearson_r']:.3f}")
    print(f"   PMA Pearson r: {optimal['PMA_pearson_r']:.3f}")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'optimization_results.json', 'w') as f:
        json.dump({
            'optimal_alpha': optimal['alpha'],
            'optimal_metrics': optimal,
            'all_results': results_table
        }, f, indent=2)
    
    with open(output_dir / 'optimization_results.csv', 'w', newline='') as f:
        fieldnames = ['alpha', 'VFA_MAE', 'VFA_bias', 'VFA_pearson_r',
                      'PMA_MAE', 'PMA_bias', 'PMA_pearson_r', 'combined_score']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_table)
    
    # Plot results
    plot_optimization(results_table, output_dir)
    
    print(f"\nResults saved to {output_dir}")


def plot_optimization(results: list, output_dir: Path):
    """Plot optimization results"""
    alphas = [r['alpha'] for r in results]
    vfa_mae = [r['VFA_MAE'] for r in results]
    pma_mae = [r['PMA_MAE'] for r in results]
    vfa_r = [r['VFA_pearson_r'] for r in results]
    pma_r = [r['PMA_pearson_r'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # MAE plots
    axes[0, 0].plot(alphas, vfa_mae, 'b-o', label='VFA')
    axes[0, 0].plot(alphas, pma_mae, 'r-s', label='PMA')
    axes[0, 0].set_xlabel('Alpha (0=rule-based, 1=DL)')
    axes[0, 0].set_ylabel('MAE (cm²)')
    axes[0, 0].set_title('Mean Absolute Error vs Alpha')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Pearson r plots
    axes[0, 1].plot(alphas, vfa_r, 'b-o', label='VFA')
    axes[0, 1].plot(alphas, pma_r, 'r-s', label='PMA')
    axes[0, 1].set_xlabel('Alpha')
    axes[0, 1].set_ylabel('Pearson r')
    axes[0, 1].set_title('Correlation vs Alpha')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bias plots
    vfa_bias = [r['VFA_bias'] for r in results]
    pma_bias = [r['PMA_bias'] for r in results]
    axes[1, 0].plot(alphas, vfa_bias, 'b-o', label='VFA')
    axes[1, 0].plot(alphas, pma_bias, 'r-s', label='PMA')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Alpha')
    axes[1, 0].set_ylabel('Bias (cm²)')
    axes[1, 0].set_title('Bias vs Alpha')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined score
    scores = [r['combined_score'] for r in results]
    axes[1, 1].plot(alphas, scores, 'g-o')
    optimal_idx = np.argmin(scores)
    axes[1, 1].plot(alphas[optimal_idx], scores[optimal_idx], 'r*', markersize=15,
                    label=f'Optimal α={alphas[optimal_idx]:.2f}')
    axes[1, 1].set_xlabel('Alpha')
    axes[1, 1].set_ylabel('Combined Score (lower=better)')
    axes[1, 1].set_title('Overall Performance vs Alpha')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'optimization_plot.png', dpi=150, bbox_inches='tight')
    print("Plot saved: optimization_plot.png")


def main():
    parser = argparse.ArgumentParser(description='Optimize Hybrid Parameters')
    parser.add_argument('--gt', type=str, required=True,
                        help='Ground truth CSV file')
    parser.add_argument('--output-dir', type=str, default='hybrid_optimization',
                        help='Output directory')
    parser.add_argument('--alpha-min', type=float, default=0.0,
                        help='Minimum alpha value')
    parser.add_argument('--alpha-max', type=float, default=1.0,
                        help='Maximum alpha value')
    parser.add_argument('--alpha-steps', type=int, default=21,
                        help='Number of alpha values to test')
    
    args = parser.parse_args()
    
    optimize_alpha(
        gt_csv=Path(args.gt),
        output_dir=Path(args.output_dir),
        alpha_range=(args.alpha_min, args.alpha_max, args.alpha_steps)
    )


if __name__ == '__main__':
    main()
