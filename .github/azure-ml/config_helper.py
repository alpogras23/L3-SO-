#!/usr/bin/env python3
"""
Azure ML Configuration Helper
- Workspace credentials'ı kontrol et
- Dataset'leri list/register et
- Compute status'ını göster
- Run'ları izle
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

try:
    from azureml.core import Workspace, Environment
    from azureml.core.compute import ComputeTarget, AmlCompute
    from azureml.core.experiment import Experiment
except ImportError:
    print("❌ Azure ML SDK yüklemesi gerekli: pip install azureml-sdk")
    sys.exit(1)


class AzureMLHelper:
    def __init__(self):
        try:
            self.ws = Workspace.from_config()
            print(f"✅ Workspace bağlantısı: {self.ws.name}")
        except Exception as e:
            print(f"❌ Workspace bağlantı hatası: {e}")
            sys.exit(1)

    def status(self):
        """Workspace durumunu göster"""
        print("\n📊 Azure ML Workspace Status")
        print("=" * 60)

        print(f"  Workspace: {self.ws.name}")
        print(f"  Subscription: {self.ws.subscription_id}")
        print(f"  Resource Group: {self.ws.resource_group}")
        print(f"  Location: {self.ws.location}")

        # Compute targets
        print("\n  🖥️  Compute Targets:")
        try:
            compute_targets = self.ws.compute_targets
            if not compute_targets:
                print("    (Yok)")
            else:
                for name, ct in compute_targets.items():
                    if isinstance(ct, AmlCompute):
                        print(f"    - {name}: {ct.status} ({ct.serialize()})")
                    else:
                        print(f"    - {name}: {ct.status}")
        except Exception as e:
            print(f"    ⚠️ Hata: {e}")

        # Environments
        print("\n  🐍 Environments:")
        try:
            envs = Environment.list(self.ws)
            for env in envs:
                print(f"    - {env.name} (v{env.version})")
        except Exception as e:
            print(f"    ⚠️ Hata: {e}")

        # Experiments
        print("\n  📋 Experiments:")
        try:
            experiments = Experiment.list(self.ws)
            for exp in experiments:
                print(f"    - {exp.name}")
        except Exception as e:
            print(f"    ⚠️ Hata: {e}")

    def list_runs(self, experiment_name: str = None, limit: int = 10):
        """Run'ları list et"""
        print("\n📊 Recent Runs")
        print("=" * 60)

        try:
            if experiment_name:
                exp = Experiment(self.ws, experiment_name)
                runs = list(exp.get_runs())[:limit]
                print(f"Experiment: {experiment_name}")
            else:
                # Tüm experiments'deki runs
                experiments = Experiment.list(self.ws)
                runs = []
                for exp in experiments:
                    runs.extend(list(exp.get_runs())[:5])
                runs = sorted(runs, key=lambda r: r.start_time, reverse=True)[:limit]

            for run in runs:
                status = "✅" if run.status == "Completed" else "⏳" if run.status == "Running" else "❌"
                print(
                    f"  {status} {run.experiment.name}/{run.id[:8]} ({run.status})"
                )
                print(f"      Started: {run.start_time}")
                print(f"      URL: {run.get_portal_url()}")
        except Exception as e:
            print(f"  ❌ Hata: {e}")

    def monitor_run(self, run_id: str):
        """Spesifik run'ı izle"""
        print(f"\n📊 Run Monitoring: {run_id}")
        print("=" * 60)

        try:
            # Run'ı bul
            run = None
            for exp in Experiment.list(self.ws):
                try:
                    run = Run.get(self.ws, run_id)
                    break
                except Exception:
                    continue

            if not run:
                print(f"❌ Run bulunamadı: {run_id}")
                return

            print(f"  Experiment: {run.experiment.name}")
            print(f"  Status: {run.status}")
            print(f"  Started: {run.start_time}")
            print(f"  Duration: {run.duration}")

            # Metrics
            print("\n  📈 Metrics:")
            try:
                for metric_name, metric_values in run.get_metrics().items():
                    if metric_values:
                        latest = metric_values[-1]
                        print(f"    - {metric_name}: {latest}")
            except Exception:
                print("    (Yok)")

            # Portal URL
            print(f"\n  🔗 Portal: {run.get_portal_url()}")

        except Exception as e:
            print(f"  ❌ Hata: {e}")

    def download_run(self, run_id: str, output_dir: str):
        """Run çıktılarını indir"""
        print(f"\n📥 Downloading run: {run_id}")
        print("=" * 60)

        try:
            import subprocess

            output_path = Path(output_dir) / run_id
            output_path.mkdir(exist_ok=True, parents=True)

            # Azure CLI ile indir
            subprocess.run(
                [
                    "az",
                    "ml",
                    "run",
                    "download",
                    "--run-id",
                    run_id,
                    "--output",
                    str(output_path),
                ],
                check=True,
            )

            print(f"  ✅ İndirildi: {output_path}")

            # Dosyaları list et
            files = list(output_path.rglob("*"))
            print(f"  📂 {len(files)} dosya")
            for f in files[:10]:
                print(f"    - {f.relative_to(output_path)}")

        except Exception as e:
            print(f"  ❌ Hata: {e}")

    def cleanup_config(self):
        """Workspace config'ini göster"""
        config_file = Path.cwd() / ".azureml" / "config.json"
        if config_file.exists():
            print(f"\n📄 Config: {config_file}")
            with open(config_file) as f:
                config = json.load(f)
                print(json.dumps(config, indent=2))
        else:
            print(f"⚠️ Config file not found: {config_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Azure ML Configuration Helper"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Status command
    subparsers.add_parser("status", help="Show workspace status")

    # List runs command
    list_runs_parser = subparsers.add_parser("list_runs", help="List recent runs")
    list_runs_parser.add_argument(
        "--experiment", type=str, help="Filter by experiment name"
    )
    list_runs_parser.add_argument("--limit", type=int, default=10, help="Limit")

    # Monitor run command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor a specific run")
    monitor_parser.add_argument("run_id", type=str, help="Run ID")

    # Download run command
    download_parser = subparsers.add_parser("download", help="Download run outputs")
    download_parser.add_argument("run_id", type=str, help="Run ID")
    download_parser.add_argument(
        "--output", type=str, default="./azure_ml_downloads", help="Output directory"
    )

    # Config command
    subparsers.add_parser("config", help="Show workspace config")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    helper = AzureMLHelper()

    if args.command == "status":
        helper.status()
    elif args.command == "list_runs":
        helper.list_runs(args.experiment, args.limit)
    elif args.command == "monitor":
        helper.monitor_run(args.run_id)
    elif args.command == "download":
        helper.download_run(args.run_id, args.output)
    elif args.command == "config":
        helper.cleanup_config()


if __name__ == "__main__":
    main()
