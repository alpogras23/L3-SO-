#!/usr/bin/env python3
"""
Azure ML: Generate Final Analysis Report
- Rule-based ve hybrid VFA/PMA sonuçlarını karşılaştır
- Training metrikleri dahil et
- Summary statistics ve vizualizasyonlar
"""

import json
from pathlib import Path
from typing import Dict
import statistics

import numpy as np
import matplotlib.pyplot as plt


class ReportGenerator:
    """Final report generation"""

    def __init__(self, rule_based_dir: Path, hybrid_dir: Path, model_dir: Path, output_dir: Path):
        self.rule_based_dir = rule_based_dir
        self.hybrid_dir = hybrid_dir
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def load_results(self, results_dir: Path) -> Dict:
        """Load all case results from directory"""
        all_results = {}

        for result_file in results_dir.glob("*_results.json"):
            case_id = result_file.stem.replace("_results", "")
            with open(result_file, "r") as f:
                all_results[case_id] = json.load(f)

        return all_results

    def compute_statistics(self, results: Dict) -> Dict:
        """Compute statistics from results"""
        successful = [r for r in results.values() if r.get("status") == "success"]

        if not successful:
            return {"total": len(results), "successful": 0}

        vfa_values = [r["VFA_cm2"] for r in successful]
        pma_values = [r["PMA_cm2"] for r in successful]
        ratio_values = [r["VFA_PMA_ratio"] for r in successful if r.get("VFA_PMA_ratio")]

        stats = {
            "total": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "vfa_cm2": {
                "mean": np.mean(vfa_values),
                "std": np.std(vfa_values),
                "min": np.min(vfa_values),
                "max": np.max(vfa_values),
            },
            "pma_cm2": {
                "mean": np.mean(pma_values),
                "std": np.std(pma_values),
                "min": np.min(pma_values),
                "max": np.max(pma_values),
            },
            "ratio": {
                "mean": np.mean(ratio_values) if ratio_values else None,
                "std": np.std(ratio_values) if ratio_values else None,
            },
        }

        return stats

    def compare_methods(self, rule_based: Dict, hybrid: Dict) -> Dict:
        """Compare rule-based vs hybrid results"""
        comparison = {"cases": {}}

        for case_id in rule_based:
            if case_id not in hybrid:
                continue

            rule = rule_based[case_id]
            hyb = hybrid[case_id]

            if rule.get("status") != "success" or hyb.get("status") != "success":
                continue

            diff_vfa = hyb["VFA_cm2"] - rule["VFA_cm2"]
            diff_pma = hyb["PMA_cm2"] - rule["PMA_cm2"]
            pct_change_vfa = (diff_vfa / rule["VFA_cm2"] * 100) if rule["VFA_cm2"] > 0 else 0
            pct_change_pma = (diff_pma / rule["PMA_cm2"] * 100) if rule["PMA_cm2"] > 0 else 0

            comparison["cases"][case_id] = {
                "rule_based_vfa_cm2": rule["VFA_cm2"],
                "hybrid_vfa_cm2": hyb["VFA_cm2"],
                "vfa_diff_cm2": diff_vfa,
                "vfa_pct_change": pct_change_vfa,
                "rule_based_pma_cm2": rule["PMA_cm2"],
                "hybrid_pma_cm2": hyb["PMA_cm2"],
                "pma_diff_cm2": diff_pma,
                "pma_pct_change": pct_change_pma,
            }

        # Summary
        if comparison["cases"]:
            vfa_diffs = [c["vfa_diff_cm2"] for c in comparison["cases"].values()]
            pma_diffs = [c["pma_diff_cm2"] for c in comparison["cases"].values()]

            comparison["summary"] = {
                "cases_compared": len(comparison["cases"]),
                "avg_vfa_diff_cm2": np.mean(vfa_diffs),
                "avg_pma_diff_cm2": np.mean(pma_diffs),
                "avg_vfa_pct_change": np.mean([c["vfa_pct_change"] for c in comparison["cases"].values()]),
                "avg_pma_pct_change": np.mean([c["pma_pct_change"] for c in comparison["cases"].values()]),
            }

        return comparison

    def load_training_metrics(self) -> Dict:
        """Load training history"""
        history_file = self.model_dir / "training_history.json"

        if not history_file.exists():
            return {}

        with open(history_file, "r") as f:
            history = json.load(f)

        return {
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "best_val_loss": min(history["val_loss"]) if history["val_loss"] else None,
            "num_epochs": len(history["train_loss"]),
        }

    def create_visualizations(self):
        """Create comparison plots"""
        rule_based = self.load_results(self.rule_based_dir)
        hybrid = self.load_results(self.hybrid_dir)
        comparison = self.compare_methods(rule_based, hybrid)

        if not comparison["cases"]:
            print("⚠️ Karşılaştırmak için yeterli başarılı vaka yok")
            return

        case_ids = list(comparison["cases"].keys())
        rule_vfa = [comparison["cases"][c]["rule_based_vfa_cm2"] for c in case_ids]
        hybrid_vfa = [comparison["cases"][c]["hybrid_vfa_cm2"] for c in case_ids]
        rule_pma = [comparison["cases"][c]["rule_based_pma_cm2"] for c in case_ids]
        hybrid_pma = [comparison["cases"][c]["hybrid_pma_cm2"] for c in case_ids]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # VFA comparison
        axes[0, 0].scatter(rule_vfa, hybrid_vfa, alpha=0.6)
        axes[0, 0].plot([min(rule_vfa), max(rule_vfa)], [min(rule_vfa), max(rule_vfa)], "r--")
        axes[0, 0].set_xlabel("Rule-based VFA (cm²)")
        axes[0, 0].set_ylabel("Hybrid VFA (cm²)")
        axes[0, 0].set_title("VFA Comparison")
        axes[0, 0].grid(True, alpha=0.3)

        # PMA comparison
        axes[0, 1].scatter(rule_pma, hybrid_pma, alpha=0.6, color="orange")
        axes[0, 1].plot([min(rule_pma), max(rule_pma)], [min(rule_pma), max(rule_pma)], "r--")
        axes[0, 1].set_xlabel("Rule-based PMA (cm²)")
        axes[0, 1].set_ylabel("Hybrid PMA (cm²)")
        axes[0, 1].set_title("PMA Comparison")
        axes[0, 1].grid(True, alpha=0.3)

        # VFA difference distribution
        vfa_diffs = [comparison["cases"][c]["vfa_pct_change"] for c in case_ids]
        axes[1, 0].hist(vfa_diffs, bins=20, alpha=0.7, edgecolor="black")
        axes[1, 0].set_xlabel("VFA % Change (Hybrid - Rule-based)")
        axes[1, 0].set_title(f"VFA Change Distribution (mean: {np.mean(vfa_diffs):.2f}%)")
        axes[1, 0].grid(True, alpha=0.3, axis="y")

        # PMA difference distribution
        pma_diffs = [comparison["cases"][c]["pma_pct_change"] for c in case_ids]
        axes[1, 1].hist(pma_diffs, bins=20, alpha=0.7, color="orange", edgecolor="black")
        axes[1, 1].set_xlabel("PMA % Change (Hybrid - Rule-based)")
        axes[1, 1].set_title(f"PMA Change Distribution (mean: {np.mean(pma_diffs):.2f}%)")
        axes[1, 1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plot_path = self.output_dir / "comparison_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"📊 Plot saved: {plot_path}")

    def generate(self):
        """Generate complete report"""
        print("=" * 60)
        print("📊 Generating final analysis report...")
        print("=" * 60)

        # Load results
        rule_based = self.load_results(self.rule_based_dir)
        hybrid = self.load_results(self.hybrid_dir)

        # Compute stats
        rule_stats = self.compute_statistics(rule_based)
        hybrid_stats = self.compute_statistics(hybrid)

        # Compare
        comparison = self.compare_methods(rule_based, hybrid)

        # Load training metrics
        training_metrics = self.load_training_metrics()

        # Compile report
        report = {
            "title": "L3 VFA/PMA Analysis Report",
            "rule_based_statistics": rule_stats,
            "hybrid_statistics": hybrid_stats,
            "method_comparison": comparison.get("summary", {}),
            "training_metrics": training_metrics,
            "timestamp": str(Path.cwd()),
        }

        # Save report
        report_path = self.output_dir / "final_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"✅ Report saved: {report_path}")

        # Generate visualizations
        self.create_visualizations()

        # Print summary
        print("\n" + "=" * 60)
        print("📈 ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"\nRule-based:")
        print(f"  Successful cases: {rule_stats['successful']}/{rule_stats['total']}")
        if rule_stats["successful"] > 0:
            print(f"  VFA: {rule_stats['vfa_cm2']['mean']:.1f} ± {rule_stats['vfa_cm2']['std']:.1f} cm²")
            print(f"  PMA: {rule_stats['pma_cm2']['mean']:.1f} ± {rule_stats['pma_cm2']['std']:.1f} cm²")

        print(f"\nHybrid (DL-enhanced):")
        print(f"  Successful cases: {hybrid_stats['successful']}/{hybrid_stats['total']}")
        if hybrid_stats["successful"] > 0:
            print(f"  VFA: {hybrid_stats['vfa_cm2']['mean']:.1f} ± {hybrid_stats['vfa_cm2']['std']:.1f} cm²")
            print(f"  PMA: {hybrid_stats['pma_cm2']['mean']:.1f} ± {hybrid_stats['pma_cm2']['std']:.1f} cm²")

        if comparison.get("summary"):
            print(f"\nComparison (Hybrid vs Rule-based):")
            print(f"  Cases compared: {comparison['summary']['cases_compared']}")
            print(f"  Avg VFA change: {comparison['summary']['avg_vfa_pct_change']:.2f}%")
            print(f"  Avg PMA change: {comparison['summary']['avg_pma_pct_change']:.2f}%")

        if training_metrics:
            print(f"\nTraining Metrics:")
            print(f"  Epochs: {training_metrics.get('num_epochs', 'N/A')}")
            print(f"  Best val loss: {training_metrics.get('best_val_loss', 'N/A')}")

        print("\n" + "=" * 60)


def main(args):
    generator = ReportGenerator(
        Path(args.rule_based),
        Path(args.hybrid),
        Path(args.model_dir),
        Path(args.output_dir),
    )
    generator.generate()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate final analysis report")
    parser.add_argument("--rule_based", type=str, required=True, help="Rule-based results directory")
    parser.add_argument("--hybrid", type=str, required=True, help="Hybrid results directory")
    parser.add_argument("--model_dir", type=str, required=True, help="Model directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    args = parser.parse_args()
    main(args)
