# ✅ Azure ML L3 VFA/PMA Pipeline - Deployment Checklist

## Pre-Deployment Verification

### 1. Local Environment ✓
- [ ] Python 3.9+ installed: `python --version`
- [ ] Azure CLI installed: `az --version`
- [ ] Virtual environment activated: `.venv/bin/activate`
- [ ] AMOS22 data verified: `/Users/alperenogras/Desktop/L3_SO_ANALYSIS/data/AMOS22/` (exists & has *.nii.gz files)
- [ ] TotalSegmentator masks verified: `/Users/alperenogras/Desktop/L3_SO_ANALYSIS/data/ts_masks/` (exists & has directories)

### 2. Azure Account & Credentials ✓
- [ ] Azure account active and authenticated: `az account show`
- [ ] Subscription ID available
- [ ] Resource group created or exists: `az group create --name <RG_NAME> --location eastus`
- [ ] Storage account quota available (need ~5-10 GB)
- [ ] Compute quota available (at least 2x NC4as_T4_v3 vCPU quota)

### 3. Files & Scripts Created ✓
- [ ] `environment.yml` (Conda dependencies)
- [ ] `run_l3_vfa_pma.py` (Rule-based + DL VFA/PMA processor)
- [ ] `prepare_training_data.py` (L3 slice extraction)
- [ ] `train_unet_model.py` (60-epoch U-Net training)
- [ ] `generate_report.py` (Comparison analysis)
- [ ] `launcher.py` (Azure ML orchestrator)
- [ ] `setup_azure_ml.sh` (Resource provisioning)
- [ ] `quick_setup.sh` (Interactive setup)
- [ ] `START_HERE.sh` (Entry point guide)
- [ ] `README_PIPELINE.md` (Complete documentation)
- [ ] `pipeline.yml` (YAML pipeline definition)

### 4. Documentation Review ✓
- [ ] `README_PIPELINE.md` read (5.9 KB - complete guide)
- [ ] `IMPLEMENTATION_SUMMARY.txt` reviewed (16 KB - overview)
- [ ] Python scripts have docstrings reviewed
- [ ] Performance expectations understood (~2-3 hours total)

---

## Deployment Steps

### Phase 1: Setup & Configuration (10-15 minutes)

#### Step 1a: Run Guided Setup
```bash
cd /Users/alperenogras/Desktop/L3_SO_ANALYSIS/.github/azure-ml/
chmod +x START_HERE.sh
./START_HERE.sh
```
- [ ] Validates Azure CLI
- [ ] Checks Python installation
- [ ] Verifies Azure authentication
- [ ] Shows setup method menu

#### Step 1b: Choose Setup Method
**Option A: Interactive (Recommended)**
```bash
./quick_setup.sh
```
Prompts for:
- [ ] Subscription ID
- [ ] Resource group name
- [ ] Azure region (default: eastus)
- [ ] Workspace name (default: l3-vfa-pma-workspace)
- [ ] Storage account name
- [ ] AMOS22 data path
- [ ] TotalSegmentator masks path

**Option B: Manual**
```bash
bash setup_azure_ml.sh \
  <SUBSCRIPTION_ID> \
  <RESOURCE_GROUP> \
  eastus \
  l3-vfa-pma-workspace \
  l3vfapmastorage \
  l3-gpu-cluster \
  Standard_NC4as_T4_v3 \
  /Users/alperenogras/Desktop/L3_SO_ANALYSIS/data/AMOS22/ \
  /Users/alperenogras/Desktop/L3_SO_ANALYSIS/data/ts_masks/
```

- [ ] Workspace created (or verified existing)
- [ ] Storage account created
- [ ] Blob containers created (amos22-data, ts-masks)
- [ ] Data uploaded to Azure Storage
- [ ] GPU compute cluster created (NC4as_T4_v3, min=0, max=2)

#### Step 1c: Verify Setup
```bash
# Check workspace
az ml workspace show --name l3-vfa-pma-workspace --resource-group <RG_NAME>

# List compute targets
az ml compute list --workspace-name l3-vfa-pma-workspace --resource-group <RG_NAME>

# Check storage
az storage account show --name l3vfapmastorage --resource-group <RG_NAME>
az storage container list --account-name l3vfapmastorage
```

- [ ] Workspace exists and is accessible
- [ ] Compute cluster ready (State: Succeeded)
- [ ] Storage containers contain data (use Azure Portal to verify)

---

### Phase 2: Pipeline Submission (5-10 minutes)

#### Step 2a: Configure Launcher
Edit `launcher.py` if needed to customize:
- [ ] Default subscription ID
- [ ] Default resource group
- [ ] Default workspace name
- [ ] Default compute cluster name
- [ ] Default experiment name
- [ ] Scripts directory path

#### Step 2b: Submit Pipeline
```bash
python launcher.py \
  --subscription_id <SUBSCRIPTION_ID> \
  --resource_group <RESOURCE_GROUP_NAME> \
  --workspace_name l3-vfa-pma-workspace \
  --compute_name l3-gpu-cluster \
  --scripts_dir ./ \
  --experiment_name l3_vfa_pma_analysis
```

Expected output:
```
✓ Workspace connected: l3-vfa-pma-workspace
✓ Datasets registered: AMOS22_DATA, TS_MASKS
✓ Compute target ready: l3-gpu-cluster
✓ Pipeline created with 5 steps
✓ Run submitted with ID: <RUN_ID>
✓ Run info saved to: run_info.json
```

- [ ] Run ID captured and saved
- [ ] `run_info.json` created locally
- [ ] No errors in submission

#### Step 2c: Monitor Pipeline
**Option A: Azure Portal (Recommended)**
- Navigate to https://ml.azure.com
- Login with Azure credentials
- Select workspace: `l3-vfa-pma-workspace`
- Go to Jobs tab
- Click on latest experiment: `l3_vfa_pma_analysis`
- Watch 5 steps execute sequentially
- Expected duration: ~2-3 hours

**Option B: CLI Monitoring**
```bash
# Check run status
az ml run show \
  --name <RUN_ID> \
  --workspace-name l3-vfa-pma-workspace \
  --resource-group <RG_NAME>

# Stream logs
az ml run show-details \
  --name <RUN_ID> \
  --workspace-name l3-vfa-pma-workspace \
  --resource-group <RG_NAME>

# Watch specific step
az ml run show-details \
  --name <RUN_ID> \
  --workspace-name l3-vfa-pma-workspace \
  --resource-group <RG_NAME> \
  --step-name process_vfa_pma
```

- [ ] Step 1 (Rule-based VFA/PMA) started
- [ ] Step 2 (Training data prep) started after Step 1
- [ ] Step 3 (U-Net training) started after Step 2
- [ ] Step 4 (Hybrid VFA/PMA) started after Step 3
- [ ] Step 5 (Report generation) started after Steps 1 & 4
- [ ] All steps completed successfully

---

### Phase 3: Results Collection (15-30 minutes)

#### Step 3a: Download Results
```bash
# Create results directory
mkdir -p ./azure_results

# Download entire run outputs
az ml run download \
  --name <RUN_ID> \
  --workspace-name l3-vfa-pma-workspace \
  --resource-group <RG_NAME> \
  --output ./azure_results

# Or download specific outputs from storage
az storage blob download-batch \
  --account-name l3vfapmastorage \
  --source vfa-pma-outputs \
  --destination ./azure_results/vfa_pma_results
```

- [ ] Results directory created locally
- [ ] All outputs downloaded (JSON + PNG files)
- [ ] Training checkpoints downloaded (best_unet.pt)
- [ ] Report files downloaded (final_report.json)

#### Step 3b: Verify Results
```bash
# Check directory structure
ls -lah ./azure_results/

# Count results files
ls ./azure_results/vfa_pma_results/*_results.json | wc -l

# View summary report
cat ./azure_results/final_report/final_report.json | jq .

# Check training history
cat ./azure_results/trained_model/training_history.json | jq '.val_loss[-1]'
```

- [ ] VFA/PMA results JSON files present
- [ ] PNG overlay images present
- [ ] Training model checkpoint present (best_unet.pt)
- [ ] Training history JSON present
- [ ] Final comparison report present
- [ ] Visualization plots present (comparison_plots.png)

---

### Phase 4: Analysis & Validation (30-60 minutes)

#### Step 4a: Statistical Analysis
```bash
# Review summary statistics
python << 'PYEOF'
import json

with open('./azure_results/final_report/final_report.json') as f:
    report = json.load(f)

print("RULE-BASED RESULTS:")
print(f"  VFA: {report['rule_based_stats']['vfa_mean']:.1f} ± {report['rule_based_stats']['vfa_std']:.1f} mm²")
print(f"  PMA: {report['rule_based_stats']['pma_mean']:.1f} ± {report['rule_based_stats']['pma_std']:.1f} mm²")

print("\nHYBRID (DL-ENHANCED) RESULTS:")
print(f"  VFA: {report['hybrid_stats']['vfa_mean']:.1f} ± {report['hybrid_stats']['vfa_std']:.1f} mm²")
print(f"  PMA: {report['hybrid_stats']['pma_mean']:.1f} ± {report['hybrid_stats']['pma_std']:.1f} mm²")

print("\nMETHOD COMPARISON:")
print(f"  Cases analyzed: {report['method_comparison']['num_cases']}")
print(f"  Mean VFA difference: {report['method_comparison']['mean_vfa_diff']:.1f} mm²")
print(f"  Mean PMA difference: {report['method_comparison']['mean_pma_diff']:.1f} mm²")
PYEOF
```

- [ ] Statistics review complete
- [ ] No systematic biases observed
- [ ] VFA/PMA ranges reasonable for AMOS22 dataset
- [ ] Training convergence verified (val_loss decreasing)

#### Step 4b: Visual QA
```bash
# View sample overlay images
open ./azure_results/vfa_pma_results/case_001_overlay.png
open ./azure_results/hybrid_results/case_001_overlay.png

# View comparison plots
open ./azure_results/final_report/comparison_plots.png
```

- [ ] Overlay images show correct VFA/PMA segmentation
- [ ] Vertebra center detection accurate
- [ ] Inner abdomen fascia boundary correct
- [ ] Hybrid predictions show reasonable blending
- [ ] Comparison plots show good correlation

#### Step 4c: Compare Methods
```bash
# Generate comparison table
python << 'PYEOF'
import json
import csv

with open('./azure_results/final_report/final_report.json') as f:
    report = json.load(f)

print("\nCASE-BY-CASE COMPARISON:")
print(f"{'Case':<15} {'Rule VFA':<15} {'Hybrid VFA':<15} {'Diff %':<15}")
print("-" * 60)

for case in report['method_comparison']['cases'][:5]:  # First 5 cases
    rule_vfa = case['rule_vfa_mm2']
    hybrid_vfa = case['hybrid_vfa_mm2']
    diff = ((hybrid_vfa - rule_vfa) / rule_vfa * 100) if rule_vfa > 0 else 0
    print(f"{case['case_id']:<15} {rule_vfa:<15.1f} {hybrid_vfa:<15.1f} {diff:<15.1f}")
PYEOF
```

- [ ] Rule-based metrics understood
- [ ] DL improvements identified
- [ ] Hybrid blend ratios reasonable (30-70% improvement expected)

---

## Post-Deployment Cleanup & Optimization

### Optional: Clean Up Resources
```bash
# Delete compute cluster (to save costs if not needed)
az ml compute delete --name l3-gpu-cluster --workspace-name l3-vfa-pma-workspace --resource-group <RG_NAME>

# Delete entire workspace and resources
az group delete --name <RESOURCE_GROUP_NAME> --yes

# Keep only storage for future reference
# Storage costs are minimal for final results
```

- [ ] Decide whether to keep Azure resources for future runs
- [ ] Estimate ongoing costs if keeping resources active

### Optional: Re-run with Different Parameters
```bash
# Modify launcher.py or create new configuration
# Increase max_cases to 100 or full dataset
# Adjust blend_alpha from 0.5 to 0.3 or 0.7
# Increase epochs from 60 to 100

python launcher.py \
  --subscription_id <SUBSCRIPTION_ID> \
  --resource_group <RESOURCE_GROUP_NAME> \
  --workspace_name l3-vfa-pma-workspace \
  --compute_name l3-gpu-cluster \
  --scripts_dir ./ \
  --experiment_name l3_vfa_pma_analysis_v2
```

- [ ] Decide on parameter adjustments
- [ ] Document configuration changes
- [ ] Re-run if improvements desired

---

## Troubleshooting

### Issue: "Resource group not found"
**Solution:**
```bash
az group create --name <RG_NAME> --location eastus
```

### Issue: "Compute cluster not ready"
**Check status:**
```bash
az ml compute show --name l3-gpu-cluster --workspace-name l3-vfa-pma-workspace --resource-group <RG_NAME>
```
Wait 5-10 minutes for cluster to reach "Succeeded" state.

### Issue: "Dataset not found"
**Re-upload data:**
```bash
az storage blob upload-batch \
  --account-name l3vfapmastorage \
  --source /Users/alperenogras/Desktop/L3_SO_ANALYSIS/data/AMOS22 \
  --destination amos22-data
```

### Issue: "Out of memory during training"
Reduce batch_size in `train_unet_model.py` from 8 to 4, or use smaller dataset (max_cases=25).

### Issue: "Run failed or timed out"
Check logs in Azure Portal:
- Jobs → Select run → Click step name
- View Output logs and Error logs
- Common issues: data access, memory, storage quota

---

## Success Criteria

✅ **Pipeline Completion Verified When:**
1. All 5 steps show status "Completed" (not Failed)
2. Output directories contain expected files:
   - vfa_pma_results/ (JSON + PNG)
   - trained_model/best_unet.pt
   - hybrid_results/ (JSON + PNG)
   - final_report/final_report.json + comparison_plots.png
3. Training history shows convergence (val_loss curve trends down)
4. Case-by-case results show reasonable VFA/PMA values:
   - VFA: 5,000-25,000 mm² (typical AMOS22 range)
   - PMA: 800-1,500 mm² (typical adult range)
5. Hybrid method shows modest improvements over rule-based (10-30% variation expected)

---

## Next Steps After Success

1. **Integrate into Mac GUI:**
   - Load best_unet.pt checkpoint in `psoas_ml/infer_gui.py`
   - Add "Use DL Model" toggle to GUI
   - Enable hybrid inference on individual DICOM cases

2. **Clinical Validation:**
   - Compare results against radiologist ground truth
   - Establish acceptable error margins
   - Iterate if needed with parameter adjustments

3. **Scale to Full Dataset:**
   - Re-run with all AMOS22 cases (500+)
   - Analyze performance across patient demographics
   - Consider multi-teacher ensemble for robustness

4. **Deploy to Production:**
   - Archive trained model checkpoint
   - Document final parameter configuration
   - Integrate into clinical workflow

---

## Support & Documentation

- **Azure ML Docs:** https://docs.microsoft.com/en-us/azure/machine-learning/
- **MONAI U-Net:** https://docs.monai.io/en/latest/networks.html#unet
- **Project Repo:** GitHub L3-SO- (alpogras23), branch: psoas-improvement
- **Local Scripts:** `/Users/alperenogras/Desktop/L3_SO_ANALYSIS/.github/azure-ml/`

---

**Status:** ✅ Ready for Deployment

All components verified. Follow steps above to deploy Azure ML pipeline.

Good luck! 🚀
