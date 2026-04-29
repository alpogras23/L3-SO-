#!/usr/bin/env python3
"""
Azure ML Pipeline Launcher
- Workspace ve compute'a bağlan
- Dataset'leri kaydet
- Pipeline'ı submit et
- Sonuçları izle
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from azureml.core import Workspace, Datastore, Dataset, Environment, Experiment, ScriptRunConfig
from azureml.core.compute import ComputeTarget
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration


def create_or_get_workspace(subscription_id: str, resource_group: str, workspace_name: str):
    """Get or create Azure ML workspace"""
    try:
        ws = Workspace(subscription_id, resource_group, workspace_name)
        print(f"✅ Workspace bağlantısı kuruldu: {workspace_name}")
        return ws
    except Exception as e:
        print(f"❌ Workspace bağlantı hatası: {e}")
        raise


def upload_amos_dataset(ws: Workspace, amos_local_path: Path, dataset_name: str = "amos22_dataset"):
    """Upload AMOS22 data to workspace default datastore"""
    print(f"\n📤 AMOS22 verisi Azure ML'e yükleniyor...")

    datastore = ws.get_default_datastore()

    try:
        # Mevcut dataset'i kontrol et
        existing = Dataset.get_by_name(ws, dataset_name)
        print(f"✅ Dataset zaten mevcut: {dataset_name}")
        return existing
    except:
        pass

    # Yeni dataset oluştur
    dataset = Dataset.File.from_files((datastore, "amos22-data/"))
    dataset.register(ws, name=dataset_name, create_new_version=True)

    print(f"✅ Dataset kaydedildi: {dataset_name}")
    return dataset


def setup_compute(ws: Workspace, compute_name: str = "l3-gpu-cluster"):
    """Get or create compute cluster"""
    try:
        compute = ComputeTarget(workspace=ws, name=compute_name)
        print(f"✅ Compute cluster bulundu: {compute_name}")
        return compute
    except Exception as e:
        print(f"❌ Compute cluster bulunamadı: {e}")
        raise


def create_pipeline(ws: Workspace, script_dir: Path, compute: ComputeTarget, datasets: dict):
    """Create Azure ML pipeline"""
    print("\n🔧 Pipeline oluşturuluyor...")

    # Environment
    env = Environment.from_conda_specification(
        name="l3-vfa-pma-env",
        file_path=str(script_dir / "environment.yml"),
    )

    # Run config
    run_config = RunConfiguration()
    run_config.environment = env
    run_config.target = compute

    # Pipeline veri outputs
    vfa_pma_output = PipelineData("vfa_pma_output", datastore=ws.get_default_datastore())
    training_data_output = PipelineData("training_data_output", datastore=ws.get_default_datastore())
    model_output = PipelineData("model_output", datastore=ws.get_default_datastore())
    hybrid_results_output = PipelineData("hybrid_results_output", datastore=ws.get_default_datastore())
    report_output = PipelineData("report_output", datastore=ws.get_default_datastore())

    # Step 1: Process VFA/PMA (rule-based)
    step1 = PythonScriptStep(
        name="Process VFA/PMA (Rule-based)",
        script_name="run_l3_vfa_pma.py",
        arguments=[
            "--input_data",
            datasets["amos22"].as_mount(),
            "--output_dir",
            vfa_pma_output,
            "--use_dl",
            "False",
        ],
        compute_target=compute,
        runconfig=run_config,
        source_directory=str(script_dir),
        allow_reuse=False,
    )

    # Step 2: Prepare training data
    step2 = PythonScriptStep(
        name="Prepare Training Data",
        script_name="prepare_training_data.py",
        arguments=[
            "--amos_root",
            datasets["amos22"].as_mount(),
            "--ts_root",
            datasets["ts_masks"].as_mount(),
            "--output_dir",
            training_data_output,
            "--max_cases",
            "50",
        ],
        compute_target=compute,
        runconfig=run_config,
        source_directory=str(script_dir),
        allow_reuse=False,
    )

    # Step 3: Train U-Net
    step3 = PythonScriptStep(
        name="Train U-Net Model",
        script_name="train_unet_model.py",
        arguments=[
            "--train_data",
            training_data_output,
            "--output_dir",
            model_output,
            "--epochs",
            "60",
            "--batch_size",
            "8",
            "--learning_rate",
            "1e-4",
        ],
        compute_target=compute,
        runconfig=run_config,
        source_directory=str(script_dir),
        allow_reuse=False,
        inputs=[training_data_output],
    )

    # Step 4: Process VFA/PMA with DL (hybrid)
    step4 = PythonScriptStep(
        name="Process VFA/PMA with DL (Hybrid)",
        script_name="run_l3_vfa_pma.py",
        arguments=[
            "--input_data",
            datasets["amos22"].as_mount(),
            "--output_dir",
            hybrid_results_output,
            "--use_dl",
            "True",
            "--model_path",
            f"{model_output}/best_unet.pt",
            "--blend_alpha",
            "0.5",
        ],
        compute_target=compute,
        runconfig=run_config,
        source_directory=str(script_dir),
        allow_reuse=False,
        inputs=[model_output],
    )

    # Step 5: Generate report
    step5 = PythonScriptStep(
        name="Generate Report",
        script_name="generate_report.py",
        arguments=[
            "--rule_based",
            vfa_pma_output,
            "--hybrid",
            hybrid_results_output,
            "--model_dir",
            model_output,
            "--output_dir",
            report_output,
        ],
        compute_target=compute,
        runconfig=run_config,
        source_directory=str(script_dir),
        allow_reuse=False,
        inputs=[vfa_pma_output, hybrid_results_output, model_output],
    )

    # Create pipeline
    pipeline = Pipeline(ws, steps=[step1, step2, step3, step4, step5])
    pipeline.validate()
    print("✅ Pipeline oluşturuldu ve valide edildi")

    return pipeline, report_output


def submit_pipeline(ws: Workspace, pipeline: Pipeline, experiment_name: str = "l3_vfa_pma_analysis"):
    """Submit pipeline run"""
    print(f"\n🚀 Pipeline submit ediliyor: {experiment_name}")

    experiment = Experiment(ws, experiment_name)
    run = experiment.submit(pipeline, wait_for_completion=False)

    print(f"✅ Pipeline submitted!")
    print(f"   Run ID: {run.id}")
    print(f"   Status: {run.status}")
    print(f"\n📊 Run'ı izlemek için:")
    print(f"   Azure Portal: {run.get_portal_url()}")

    return run


def main(args):
    print("=" * 60)
    print("🚀 Azure ML L3 VFA/PMA Pipeline Launcher")
    print("=" * 60)

    # Workspace
    ws = create_or_get_workspace(args.subscription_id, args.resource_group, args.workspace_name)

    # Compute
    compute = setup_compute(ws, args.compute_name)

    # Datasets
    print("\n📂 Datasets hazırlanıyor...")
    datasets = {
        "amos22": Dataset.get_by_name(ws, "amos22_dataset"),
        "ts_masks": Dataset.get_by_name(ws, "ts_masks_dataset"),
    }
    print("✅ Datasets yüklendi")

    # Pipeline
    script_dir = Path(args.scripts_dir)
    pipeline, report_output = create_pipeline(ws, script_dir, compute, datasets)

    # Submit
    run = submit_pipeline(ws, pipeline, args.experiment_name)

    # Save run info
    run_info = {
        "run_id": run.id,
        "experiment": args.experiment_name,
        "timestamp": datetime.now().isoformat(),
        "workspace": args.workspace_name,
        "compute": args.compute_name,
    }

    run_info_path = Path(args.output_dir) / "run_info.json"
    run_info_path.parent.mkdir(exist_ok=True, parents=True)
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"\n💾 Run info saved: {run_info_path}")
    print("\n⏳ Pipeline çalışıyor... Lütfen Azure Portal'da ilerlemesini izleyin.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch L3 VFA/PMA Azure ML pipeline")
    parser.add_argument("--subscription_id", type=str, required=True, help="Azure subscription ID")
    parser.add_argument("--resource_group", type=str, required=True, help="Azure resource group")
    parser.add_argument("--workspace_name", type=str, required=True, help="Azure ML workspace name")
    parser.add_argument("--compute_name", type=str, default="l3-gpu-cluster", help="Compute cluster name")
    parser.add_argument("--scripts_dir", type=str, required=True, help="Directory containing pipeline scripts")
    parser.add_argument("--experiment_name", type=str, default="l3_vfa_pma_analysis", help="Experiment name")
    parser.add_argument("--output_dir", type=str, default="./azure-ml-runs", help="Output directory for run info")

    args = parser.parse_args()
    main(args)
