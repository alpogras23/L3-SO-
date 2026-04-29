# Step 3: VFA/PMA Inference Pipeline - Complete Guide

## 🎯 Overview

**Step 3** performs HU-calibrated inference on all 240 AMOS cases using the trained U-Net from Step 2 to calculate:

- **PMA (Psoas Muscle Area)**: -29 to +150 HU
- **VAT (Visceral Fat Area)**: -150 to -50 HU
- **SAT (Subcutaneous Fat Area)**: -190 to -30 HU

## 📊 Input/Output

| Component | Details |
|-----------|---------|
| **Inputs** | Teacher labels (L3 HU slices) + trained U-Net checkpoint |
| **Processing** | MONAI U-Net inference + HU-based tissue classification |
| **Outputs** | VFA/PMA measurements for all 240 cases + validation report |
| **Duration** | 10-15 minutes (CPU) |

## 🔧 Components

### Core Modules

#### `hu_classifier.py` - HU Threshold Tissue Classification
Provides tissue-specific HU ranges and classification logic:

```python
from hu_classifier import HUAnalyzer

analyzer = HUAnalyzer()

# Analyze L3 slice
results = analyzer.analyze_slice(hu_slice, peritoneal_mask=peritoneal)

# Output: {'pma_mm2': float, 'vat_mm2': float, 'sat_mm2': float, ...}
```

**Key Classes**:
- `HUThresholds`: Configuration dataclass with tissue HU ranges
- `HUClassifier`: Binary mask generation from HU values
- `AreaCalculator`: Converts masks to mm² measurements
- `HUAnalyzer`: Complete pipeline (classify + calculate)

#### `step3_inference_vfa_pma.py` - Main Inference Script
Handles model loading, batch processing, and result serialization:

```bash
python step3_inference_vfa_pma.py \
    --teacher-labels-dir ./teacher_labels \
    --checkpoint ./best_unet.pth \
    --output-dir ./inference_results \
    --device cpu
```

**Key Features**:
- Loads trained U-Net checkpoint
- Iterates through all 240 cases
- Performs HU-based tissue classification
- Validates measurements (sanity checks)
- Saves results as JSON + summary

#### `validate_vfa_pma_accuracy.py` - Ground Truth Validation
Compares inference results with radiologist measurements:

```bash
python validate_vfa_pma_accuracy.py \
    --inference-results ./inference_results \
    --ground-truth ground_truth.csv \
    --output validation_report.json
```

**Metrics Calculated**:
- Mean Absolute Error (MAE)
- Bias and bias percentage
- Pearson/Spearman correlations
- Bland-Altman limits of agreement
- Clinical accuracy assessment

## 📝 HU Threshold Configuration

### User-Provided Ranges

```json
{
  "TISSUE_HU_RANGES": {
    "muscle_hu_min": -29,
    "muscle_hu_max": 150,
    "vat_hu_min": -150,
    "vat_hu_max": -50,
    "sat_hu_min": -190,
    "sat_hu_max": -30
  }
}
```

### Clinical Reference Values

| Tissue | HU Value | Purpose |
|--------|----------|---------|
| Air | -1000 | Reference (lungs) |
| Water | 0 | Reference |
| Fat | -100 | Reference |
| Muscle | +40 | PMA center |
| Bone | 200-3000 | Exclusion |

## 🚀 Execution Flow

### Local Testing (CPU)

```bash
# 1. Ensure teacher labels exist
ls teacher_labels/ | head -5

# 2. Run inference
python .github/azure-ml/step3_inference_vfa_pma.py \
    --teacher-labels-dir ./teacher_labels \
    --checkpoint ./step2_outputs/best_unet.pth \
    --output-dir ./inference_results \
    --device cpu

# 3. Monitor progress
tail -f inference_results/*.json | grep case_id

# 4. Validate against ground truth
python .github/azure-ml/validate_vfa_pma_accuracy.py \
    --inference-results ./inference_results \
    --ground-truth ground_truth_radyologist.csv \
    --output validation_report.json

# 5. Review results
cat validation_report.json
```

### Azure ML Job Submission

```bash
# Auto-submitted by orchestrator after Step 2
# Or manually:
az ml job create \
    --file .github/azure-ml/job_step3_inference.yml \
    --set display_name="Step 3: VFA/PMA Inference"

# Monitor
az ml job stream --name [JOB_ID]
```

## 📊 Output Structure

```
inference_results/
├── summary.json              # Summary statistics
├── 0001_vfa_pma.json        # Individual case results
├── 0002_vfa_pma.json
├── ...
└── 0500_vfa_pma.json        # 240 cases total

validation_report.json       # Ground truth comparison
```

### Example Result JSON

```json
{
  "case_id": "0001",
  "pma_mm2": 1245.3,
  "vat_mm2": 24567.8,
  "sat_mm2": 18934.2,
  "total_fat_mm2": 43502.0,
  "validity": {
    "is_valid": true,
    "message": "OK"
  },
  "hu_stats": {
    "hu_min": -180,
    "hu_max": 500,
    "hu_mean": 25,
    "hu_std": 45
  },
  "metadata": {
    "l3_index": 147,
    "patient_height_m": 1.75,
    "patient_sex": "M"
  }
}
```

## ✅ Success Criteria

### Clinical Accuracy Standards

| Metric | Target | Assessment |
|--------|--------|------------|
| **PMA MAE** | <50 mm² | Excellent |
| **VAT MAE** | <100 mm² | Excellent |
| **PMA Bias** | ±5% | Excellent |
| **VAT Bias** | ±10% | Excellent |
| **Pearson r** | >0.90 | Excellent |

### Quality Gates

✅ **PASS** when:
- [ ] All 240 cases processed without errors
- [ ] PMA MAE < 50 mm² (comparison with radiologist)
- [ ] VAT MAE < 100 mm² (comparison with radiologist)
- [ ] Bias within ±10% for both tissues
- [ ] Pearson r > 0.90 with ground truth
- [ ] No measurements outside biological range
- [ ] Confidence scores > 0.75 for 95%+ cases

⚠️ **BORDERLINE** when:
- MAE meets standards but bias trending high
- Edge cases requiring manual review
- Unusual anatomies detected

❌ **FAIL** when:
- Error rate > 5%
- Systematic bias > 20%
- Pearson r < 0.85
- Model underfitting detected

## 🔍 Quality Control Checks

### Automated Validation

```python
# Measurement sanity checks
PMA_MIN = 500    # mm²
PMA_MAX = 3000   # mm²
VAT_MIN = 2000   # mm²
VAT_MAX = 100000 # mm²
SAT_MIN = 1000   # mm²
SAT_MAX = 150000 # mm²

# Flag unusual cases
if not (PMA_MIN <= pma <= PMA_MAX):
    flag_case(case_id, "unusual_pma")
```

### Manual Review Process

1. **Extract flagged cases**
   ```bash
   grep "unusual\|out_of_range\|low_confidence" inference_results/*.json
   ```

2. **Visual inspection**
   - Review overlay images
   - Check L3 level correctness
   - Verify segmentation boundaries

3. **Comparison with radyologist**
   - Identify systematic discrepancies
   - Note anatomical variations
   - Document edge cases

## 🐛 Troubleshooting

### Issue: High PMA Variance

**Symptoms**: PMA measurements 5-10x different from expected

**Solution**:
1. Check vertebra detection (L3 level correctness)
2. Adjust muscle HU thresholds: Try -50 to +100 HU
3. Review peritoneal boundary accuracy
4. Increase training epochs if underfitting

### Issue: VAT Systematically Underestimated

**Symptoms**: VAT 20-50% lower than radiologist measurements

**Solution**:
1. Expand VAT HU range: Try -160 to -40 HU
2. Check peritoneal mask accuracy
3. Verify tissue classification algorithm
4. Retrain with adjusted HU thresholds

### Issue: Slow Inference

**Symptoms**: Processing taking >5 minutes per case

**Solution**:
1. Use GPU if available: `--device cuda`
2. Increase batch size: `--batch-size 16`
3. Profile bottleneck: `python -m cProfile step3_inference_vfa_pma.py`

## 📈 Performance Optimization

### Batch Processing

```python
# Default: 8 cases per batch
# For faster inference on GPU
--batch-size 32

# For limited memory (CPU)
--batch-size 4
```

### GPU Acceleration (if available)

```bash
python step3_inference_vfa_pma.py \
    --device cuda \
    --batch-size 16
```

### Parallel Processing (future)

```python
# Process multiple GPUs
--distributed --ngpus 4
```

## 📚 Integration with Previous Steps

### From Step 2 (U-Net Training)

```
Step 2 Output:
├── best_unet.pth          ← Checkpoint for Step 3
├── teacher_labels/        ← L3 slices for inference
└── training_metrics.json  ← Training history
```

### Step 3 Configuration

```yaml
inputs:
  checkpoint: ./best_unet.pth
  teacher_labels: ./teacher_labels

outputs:
  inference_results: ./inference_results
```

## 🔄 Post-Inference Workflow

### 1. Generate Validation Report

```bash
python .github/azure-ml/validate_vfa_pma_accuracy.py \
    --inference-results ./inference_results \
    --ground-truth ground_truth_radyologist.csv \
    --output validation_report.json
```

### 2. Create Summary Statistics

```bash
python scripts/summary_statistics.py \
    --input ./inference_results \
    --output summary_stats.csv
```

### 3. Generate Visualizations

```bash
python scripts/plot_results.py \
    --results ./inference_results \
    --ground-truth ground_truth_radyologist.csv \
    --output ./plots/
```

### 4. Clinical Review

- Share results with radiologists
- Get feedback on accuracy
- Identify outliers for discussion
- Document any adjustments needed

## 📋 Monitoring & Logging

### Azure ML Job Status

```bash
# Check job progress
az ml job show --name [STEP3_JOB_ID] --query status

# Stream logs
az ml job stream --name [STEP3_JOB_ID]

# Get output location
az ml job show --name [STEP3_JOB_ID] \
    --query "outputs.inference_results.path"
```

### Local Logs

```bash
# Check inference logs
tail -f /tmp/pipeline_log.txt

# Monitor specific case
grep "case_id" inference_results/*.json | head -20
```

## 🎓 Expected Clinical Values

### Typical Patient Populations

| Population | PMA (mm²) | VAT (mm²) | BMI |
|-----------|-----------|-----------|-----|
| Normal weight | 1000-1500 | 8000-15000 | 18.5-25 |
| Overweight | 1100-1600 | 15000-25000 | 25-30 |
| Obese | 1200-1700 | 25000-50000 | 30+ |

### Quality Indicators

✅ **Healthy Ranges**:
- PMA: 800-2000 mm²
- VAT: 5000-80000 mm² (varies with obesity)
- SAT: 3000-150000 mm²

⚠️ **Requires Review**:
- PMA < 500 or > 3000 mm²
- VAT < 2000 or > 100000 mm²
- Extreme SAT values

## 🚀 Next Steps

After Step 3 completion:

1. **Validate accuracy** vs. radiologist measurements
2. **Identify outliers** for manual review
3. **Fine-tune parameters** if needed
4. **Generate clinical report** with statistics
5. **Deploy to production** if accuracy meets standards
6. **Monitor longitudinal** performance over time

## 📞 Support

For issues or questions:

1. Check logs: `tail -f /tmp/pipeline_log.txt`
2. Review HU thresholds: `cat .github/azure-ml/HU_THRESHOLDS.cfg`
3. Run validation: `python validate_vfa_pma_accuracy.py --help`
4. Contact: See project README
