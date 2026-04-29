#!/usr/bin/env python3
"""
Training metrics visualization and analysis.
Usage: python scripts/visualize_training.py --runs run_mt_mini_nifti run_mt_10cases_10ep --out training_plots
"""
import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def load_metrics(run_dir):
    """Load Lightning CSV logger metrics."""
    csv_pattern = os.path.join(run_dir, "logs", "version_*", "metrics.csv")
    csv_files = glob.glob(csv_pattern)
    if not csv_files:
        print(f"[WARN] No metrics.csv found in {run_dir}")
        return None
    # Use latest version
    csv_path = sorted(csv_files)[-1]
    df = pd.read_csv(csv_path)
    return df


def plot_loss_curves(runs_data, out_path):
    """Plot train/val loss curves for multiple runs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for run_name, df in runs_data.items():
        if df is None:
            continue
        # Drop NaN epochs (sanity check rows)
        df = df.dropna(subset=['epoch'])
        
        # Train loss
        train_df = df.dropna(subset=['train_loss'])
        if not train_df.empty:
            ax.plot(train_df['epoch'], train_df['train_loss'], 
                   label=f'{run_name} (train)', marker='o', alpha=0.7)
        
        # Val loss
        val_df = df.dropna(subset=['val_loss'])
        if not val_df.empty:
            ax.plot(val_df['epoch'], val_df['val_loss'], 
                   label=f'{run_name} (val)', marker='s', alpha=0.7, linestyle='--')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (DiceCE)')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[SAVE] {out_path}")
    plt.close()


def plot_learning_rate(runs_data, out_path):
    """Plot learning rate schedule."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for run_name, df in runs_data.items():
        if df is None or 'lr-Adam' not in df.columns:
            continue
        df = df.dropna(subset=['epoch', 'lr-Adam'])
        if not df.empty:
            ax.plot(df['epoch'], df['lr-Adam'], label=run_name, marker='o', alpha=0.7)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[SAVE] {out_path}")
    plt.close()


def summary_table(runs_data):
    """Print summary statistics table."""
    print("\n=== Training Summary ===")
    print(f"{'Run':<30} {'Epochs':<8} {'Final Train Loss':<18} {'Final Val Loss':<18} {'Best Val Loss':<18}")
    print("-" * 100)
    
    for run_name, df in runs_data.items():
        if df is None:
            print(f"{run_name:<30} N/A")
            continue
        
        df = df.dropna(subset=['epoch'])
        max_epoch = int(df['epoch'].max()) if not df.empty else 0
        
        train_df = df.dropna(subset=['train_loss'])
        final_train = train_df['train_loss'].iloc[-1] if not train_df.empty else float('nan')
        
        val_df = df.dropna(subset=['val_loss'])
        final_val = val_df['val_loss'].iloc[-1] if not val_df.empty else float('nan')
        best_val = val_df['val_loss'].min() if not val_df.empty else float('nan')
        
        print(f"{run_name:<30} {max_epoch:<8} {final_train:<18.4f} {final_val:<18.4f} {best_val:<18.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True, help='Run directories to analyze')
    parser.add_argument('--out', default='training_plots', help='Output directory for plots')
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    # Load metrics
    runs_data = {}
    for run_dir in args.runs:
        if not os.path.exists(run_dir):
            print(f"[WARN] Run directory not found: {run_dir}")
            continue
        run_name = os.path.basename(run_dir)
        runs_data[run_name] = load_metrics(run_dir)
    
    if not runs_data:
        print("[ERROR] No valid runs found")
        return
    
    # Generate plots
    plot_loss_curves(runs_data, os.path.join(args.out, 'loss_curves.png'))
    plot_learning_rate(runs_data, os.path.join(args.out, 'learning_rate.png'))
    
    # Summary
    summary_table(runs_data)
    
    print(f"\n[DONE] Plots saved to: {args.out}")


if __name__ == "__main__":
    main()
