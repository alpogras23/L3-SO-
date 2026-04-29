# Step 3: VFA/PMA Inference Integration Guide

## Overview

Step 3 will use the trained U-Net from Step 2 to perform inference on all 240 AMOS cases, calculating:
- **PMA (Psoas Muscle Area)**: Using HU thresholds -29 to +150 HU
- **VAT (Visceral Fat Area)**: Using HU thresholds -150 to -50 HU  
- **SAT (Subcutaneous Fat Area)**: Using HU thresholds -190 to -30 HU

## HU Classifier Module

The `hu_classifier.py` module provides:

### Core Classes

#### `HUThresholds`
Configuration dataclass with tissue-specific HU ranges:
```python
from hu_classifier import HUThresholds

thresholds = HUThresholds()
# thresholds.MUSCLE_HU_MIN = -29
# thresholds.MUSCLE_HU_MAX = 150
# thresholds.VAT_HU_MIN = -150
# thresholds.VAT_HU_MAX = -50
# thresholds.SAT_HU_MIN = -190
# thresholds.SAT_HU_MAX = -30
```

#### `HUClassifier`
Performs tissue classification from HU slices:
```python
from hu_classifier import HUClassifier
import numpy as np

classifier = HUClassifier()

# Classify single tissue
muscle_mask = classifier.classify_muscle(hu_slice)  # Binary mask
vat_mask = classifier.classify_vat(hu_slice)
sat_mask = classifier.classify_sat(hu_slice)

# Or classify all at once
masks = classifier.classify_all(hu_slice, peritoneal_mask=peritoneal)
# Returns: {'muscle': array, 'vat': array, 'sat': array}
```

#### `AreaCalculator`
Converts masks to tissue areas (mm²):
```python
from hu_classifier import AreaCalculator

calculator = AreaCalculator(pixel_spacing_x=0.977, pixel_spacing_y=0.977)
area_mm2 = calculator.calculate_area(mask)

# Or multiple tissues
areas = calculator.calculate_areas(masks)
# Returns: {'muscle': float, 'vat': float, 'sat': float}
```

#### `HUAnalyzer`
Complete pipeline (classify + calculate):
```python
from hu_classifier import HUAnalyzer

analyzer = HUAnalyzer()
results = analyzer.analyze_slice(hu_slice, peritoneal_mask=peritoneal)
# Returns:
# {
#     'pma_mm2': float,      # Psoas muscle area
#     'vat_mm2': float,      # Visceral fat area
#     'sat_mm2': float,      # Subcutaneous fat area
#     'total_fat_mm2': float,# VAT + SAT
#     'muscle': array,       # Classification masks
#     'vat': array,
#     'sat': array
# }
```

## Integration into Step 3 Pipeline

### 1. **Load Trained Checkpoint**

```python
import torch
from monai.networks.nets import UNet

# Load checkpoint from Step 2
checkpoint = torch.load("step2_outputs/best_unet.pth")
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=3,  # muscle, vat, sat
    channels=(32, 64, 128, 256),
    strides=(2, 2, 2)
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### 2. **Process Each Case**

```python
import os
import json
import numpy as np
import nibabel as nib
from hu_classifier import HUAnalyzer

analyzer = HUAnalyzer()
cases_dir = "/path/to/teacher_labels"
results_dir = "/path/to/inference_results"

for case_id in sorted(os.listdir(cases_dir)):
    case_path = os.path.join(cases_dir, case_id)
    
    # Load L3 HU slice
    hu_path = os.path.join(case_path, "hu_slice.png")
    hu_slice = cv2.imread(hu_path, cv2.IMREAD_GRAYSCALE).astype(np.int16)
    hu_slice = hu_slice - 1024  # Convert back to HU values
    
    # Load metadata
    meta_path = os.path.join(case_path, "metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)
    
    # Infer peritoneal mask (optional - use segmentation)
    peritoneal_mask = model(torch.tensor(hu_slice).unsqueeze(0).unsqueeze(0).float())
    peritoneal_mask = peritoneal_mask.argmax(dim=1).cpu().numpy()[0]
    
    # Classify tissues and calculate areas
    results = analyzer.analyze_slice(hu_slice, peritoneal_mask=peritoneal_mask)
    
    # Save results
    output_file = os.path.join(results_dir, f"{case_id}_vfa_pma.json")
    with open(output_file, 'w') as f:
        json.dump({
            'case_id': case_id,
            'pma_mm2': float(results['pma_mm2']),
            'vat_mm2': float(results['vat_mm2']),
            'sat_mm2': float(results['sat_mm2']),
            'total_fat_mm2': float(results['total_fat_mm2']),
            'metadata': metadata
        }, f, indent=2)
```

### 3. **Quality Control Checks**

```python
def validate_measurements(pma_mm2: float, vat_mm2: float, sat_mm2: float) -> bool:
    """Sanity checks for VFA/PMA measurements."""
    
    # PMA range: 500-3000 mm² (typical adult)
    if not (500 <= pma_mm2 <= 3000):
        print(f"⚠️ Unusual PMA: {pma_mm2:.0f} mm² (expected 500-3000)")
        return False
    
    # VAT range: 5000-50000 mm² (depends on obesity)
    if not (2000 <= vat_mm2 <= 100000):
        print(f"⚠️ Unusual VAT: {vat_mm2:.0f} mm² (expected 2000-100000)")
        return False
    
    # SAT range: 5000-100000 mm²
    if not (1000 <= sat_mm2 <= 150000):
        print(f"⚠️ Unusual SAT: {sat_mm2:.0f} mm² (expected 1000-150000)")
        return False
    
    # PMA should be much smaller than VAT for most cases
    if pma_mm2 > vat_mm2:
        print(f"⚠️ Unusual ratio: PMA > VAT ({pma_mm2:.0f} > {vat_mm2:.0f})")
        return False
    
    return True
```

## Expected Output Structure

```
inference_results/
├── 0001_vfa_pma.json
├── 0002_vfa_pma.json
├── ...
└── 0500_vfa_pma.json

# Example JSON content:
{
  "case_id": "0001",
  "pma_mm2": 1245.3,
  "vat_mm2": 24567.8,
  "sat_mm2": 18934.2,
  "total_fat_mm2": 43502.0,
  "metadata": {
    "l3_index": 147,
    "hu_min": -180,
    "hu_max": 500,
    "hu_mean": 25,
    "patient_height_m": 1.75,
    "patient_sex": "M"
  }
}
```

## Comparison with Ground Truth (Validation)

After inference, compare with radyologist measurements:

```python
import pandas as pd
from scipy.stats import pearsonr

# Load inference results
results_df = pd.read_json("inference_results_summary.json")

# Load ground truth (from radyologist)
gt_df = pd.read_csv("ground_truth_radyologist.csv")

# Merge and compare
comparison = pd.merge(results_df, gt_df, on='case_id')

# Calculate metrics
mae_pma = (comparison['pma_mm2_inferred'] - comparison['pma_mm2_gt']).abs().mean()
mae_vat = (comparison['vat_mm2_inferred'] - comparison['vat_mm2_gt']).abs().mean()

r_pma, _ = pearsonr(comparison['pma_mm2_inferred'], comparison['pma_mm2_gt'])
r_vat, _ = pearsonr(comparison['vat_mm2_inferred'], comparison['vat_mm2_gt'])

print(f"MAE PMA: {mae_pma:.1f} mm²")
print(f"MAE VAT: {mae_vat:.1f} mm²")
print(f"Pearson r (PMA): {r_pma:.3f}")
print(f"Pearson r (VAT): {r_vat:.3f}")

# Success criteria
print("\n✅ PASS" if mae_pma < 50 and mae_vat < 100 and r_pma > 0.90 and r_vat > 0.90 else "❌ FAIL")
```

## Performance Optimization

### Batch Processing
```python
# Process multiple cases in batch
batch_size = 8
for i in range(0, len(case_ids), batch_size):
    batch_cases = case_ids[i:i+batch_size]
    hu_slices = [load_hu_slice(cid) for cid in batch_cases]
    hu_batch = np.stack(hu_slices)
    
    # Model inference
    with torch.no_grad():
        masks = model(torch.tensor(hu_batch).unsqueeze(1).float())
    
    # Process each result
    for j, case_id in enumerate(batch_cases):
        results = analyzer.analyze_slice(hu_slices[j], peritoneal_mask=masks[j])
        save_results(case_id, results)
```

### GPU Acceleration (Optional)
```python
# If GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Move batch to device
hu_batch = torch.tensor(hu_batch).unsqueeze(1).float().to(device)
with torch.no_grad():
    masks = model(hu_batch)
```

## Error Handling

```python
def safe_analyze_case(case_id: str, hu_slice: np.ndarray) -> Optional[Dict]:
    """Safely analyze case with error handling."""
    try:
        # Validation
        if hu_slice.size == 0:
            print(f"❌ Empty slice for {case_id}")
            return None
        
        if hu_slice.min() < -1000 or hu_slice.max() > 4000:
            print(f"⚠️ Unusual HU range for {case_id}: {hu_slice.min()} to {hu_slice.max()}")
        
        # Analysis
        analyzer = HUAnalyzer()
        results = analyzer.analyze_slice(hu_slice)
        
        # Validation
        if not validate_measurements(results['pma_mm2'], results['vat_mm2'], results['sat_mm2']):
            print(f"⚠️ {case_id}: Measurements outside typical range, flagged for review")
            results['qa_flag'] = 'review_needed'
        
        return results
    
    except Exception as e:
        print(f"❌ Error processing {case_id}: {e}")
        return None
```

## Timeline

| Phase | Duration | Task |
|-------|----------|------|
| Step 1 | 2-3 h | Generate teacher labels (240 cases) |
| Step 2 | 2-4 h | Train U-Net (60 epochs) |
| **Step 3** | **10-15 min** | **Inference + VFA/PMA calculation** |

## Monitoring & Logs

```bash
# Monitor Step 3 progress
tail -f /tmp/pipeline_log.txt | grep "Step 3"

# Check job status
az ml job show --name [STEP3_JOB_ID] --query status

# View results summary
python scripts/summary_results.py --input inference_results/ --out results_summary.csv
```

## Success Criteria

✅ **Pipeline Success** when all of the following are met:
- [ ] All 240 cases processed without errors
- [ ] Mean Absolute Error (MAE) < 50 mm² for PMA
- [ ] Mean Absolute Error (MAE) < 100 mm² for VAT
- [ ] Pearson correlation r > 0.90 with ground truth
- [ ] No measurements outside biological range
- [ ] Overlay images generated for visual validation

## Next Steps

After Step 3 completion:

1. **Compare with Radiologist Measurements**
   - Validate accuracy (MAE < 50 mm²)
   - Identify outlier cases for manual review

2. **Generate Summary Report**
   - Statistical comparison: mean, std, range
   - Pearson/Spearman correlations
   - Bland-Altman plot (bias and limits of agreement)

3. **Visual Quality Assurance**
   - Review overlay images for anatomically correct segmentation
   - Check for edge cases (unusual anatomy, image artifacts)

4. **Clinical Validation**
   - Ensure measurements match clinical expectations
   - Verify correct L3 level selection across all cases
   - Confirm proper tissue discrimination (muscle vs fat)

5. **Parameter Fine-Tuning** (If needed)
   - Adjust HU thresholds if systematic bias detected
   - Retrain with optimized parameters
   - Iterate until accuracy goals met
