# AMOS22 Full Automated Pipeline - Implementation Plan

## 🎯 Objective

AMOS22 abdominal BT verilerini kullanarak L3 seviyesinde **tam otomatik, hızlı ve tekrarlanabilir** VFA/PMA ölçümü yapan bir pipeline geliştirmek.

**Hedef:** Klinik kullanımda her yeni vakada TotalSegmentator/Comp2Comp çalıştırmadan, yalnızca tek bir hafif model + HU tabanlı ölçüm ile sonuç üretmek.

---

## 📊 Mimari Genel Bakış

```
┌─────────────────────────────────────────────────────────────┐
│ OFFLINE: Teacher Generation (Bir kere, AMOS22)             │
│ TotalSegmentator → Anatomik maskeler (psoas, vertebra...)  │
│ Output: teacher_labels/ (HU slices + binary masks)         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ TRAINING: Student Model (GPU, 60-100 epoch)                │
│ Input: CT HU + Teacher masks                               │
│ Output: student_model.pth (psoas_left, psoas_right, inner) │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ ONLINE: Fast Inference (Yeni vakalar, real-time)           │
│ Input: CT DICOM/NIfTI                                      │
│ Process: student_model → masks + HU thresholds            │
│ Output: VFA, PMA, PMI + QC + Overlay                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Phase 1: Offline Teacher Generation (COMPLETED ✅)

### Status: Azure ML'de çalışıyor
- ✅ Script: `.github/azure-ml/step1_real_teachers_all.py`
- ✅ TotalSegmentator task: `abdominal_muscles` (psoas segmentation)
- ✅ Output: `teacher_labels/case_*/` folders

### Teacher Outputs:
```
teacher_labels/
  case_amos_0001/
    ├── hu_slice.npy           # L3 seviyesi HU değerleri
    ├── psoas_left.npy         # Sol psoas binary mask
    ├── psoas_right.npy        # Sağ psoas binary mask
    ├── vat.npy                # Visceral fat mask (HU derived)
    ├── vertebra.npy           # Vertebra mask (L3 landmark)
    └── metadata.json          # Spacing, slice index, case info
```

### Improvements Needed:

#### 1.1 Add INNER_ABDOMEN_MASK Generation
**Problem:** Fascia boundary yok, sadece VAT var

**Solution:** `inner_abdomen_via_wall()` metodunu teacher generation'a ekle

```python
# step1_real_teachers_all.py içine eklenecek
def generate_inner_abdomen_mask(hu_slice, ds, vertebra_mask, ts_output_dir):
    """
    Generate inner abdomen mask (fascia proxy) from:
    - HU-based gradient detection
    - Vertebra anchor
    - Morphological operations
    """
    from core_mini import inner_abdomen_via_wall
    
    # Mevcut core_mini fonksiyonunu kullan
    inner, wall_edges, vb, center = inner_abdomen_via_wall(hu_slice, ds)
    
    # Posterolateral leak check
    leak_flag = check_posterolateral_leak(inner, vertebra_mask, ds)
    
    return inner, leak_flag
```

**Action Items:**
- [ ] `step1_real_teachers_all.py` → `inner_abdomen.npy` çıktısı ekle
- [ ] `core_mini.inner_abdomen_via_wall()` fonksiyonunu import et
- [ ] Leak detection QC flag'i metadata'ya ekle

#### 1.2 Improve VAT Mask Quality
**Current:** Sadece HU threshold (-190 to -30)

**Enhancement:** 
```python
def build_vat_mask(hu_slice, inner_abdomen_mask):
    """VAT = Fat HU ∩ Inner Abdomen"""
    fat_hu_mask = (hu_slice >= -190) & (hu_slice <= -30)
    vat_mask = fat_hu_mask & inner_abdomen_mask
    return vat_mask
```

---

## 🤖 Phase 2: Student Model Training (IN PROGRESS ⚙️)

### Status: Azure ML'de başlatıldı, iyileştirme gerekli

### Current Training Script:
- 📄 `.github/azure-ml/step2_train_unet.py`
- 🔧 Model: MONAI U-Net
- 📊 Output channels: 3 (psoas_left, psoas_right, vat)

### Improvements Needed:

#### 2.1 Add INNER_ABDOMEN Output Channel
**Change:** 3 → 4 output channels

```python
# step2_train_unet.py
model = UNet(
    spatial_dims=2,
    in_channels=1,         # HU slice
    out_channels=4,        # psoas_left, psoas_right, vat, inner_abdomen
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2
)
```

**Dataset Changes:**
```python
class L3TeacherDataset(Dataset):
    def __getitem__(self, idx):
        # Load labels
        psoas_left = np.load(case_dir / "psoas_left.npy")
        psoas_right = np.load(case_dir / "psoas_right.npy")
        vat = np.load(case_dir / "vat.npy")
        inner_abdomen = np.load(case_dir / "inner_abdomen.npy")  # NEW
        
        # Combined label: [psoas_left, psoas_right, vat, inner_abdomen]
        labels = torch.from_numpy(
            np.stack([psoas_left, psoas_right, vat, inner_abdomen], axis=0)
        )
        return {"image": image, "labels": labels}
```

#### 2.2 Add L3 Slice Locator (Optional but Recommended)
**Goal:** Model otomatik olarak L3 seviyesini bulsun

**Approach 1:** Vertebra heatmap regression
```python
# Add vertebra center heatmap as 5th output channel
# Training: Gaussian blob around vertebra center
# Inference: Find peak → L3 slice center
```

**Approach 2:** Slice classification head
```python
# Multi-task: Segmentation + Slice classification (L1-L5)
# Adds auxiliary loss for L3 identification
```

#### 2.3 Advanced Augmentations
```python
# Add to training transform
transforms = Compose([
    RandRotate90(prob=0.5),
    RandFlip(spatial_axis=0, prob=0.5),
    RandFlip(spatial_axis=1, prob=0.5),
    RandGaussianNoise(prob=0.3, std=0.05),
    RandAdjustContrast(prob=0.3, gamma=(0.8, 1.2)),
    # HU-specific augmentation
    RandScaleIntensity(factors=0.1, prob=0.3),  # Simulate scanner differences
])
```

#### 2.4 Loss Function Tuning
```python
# Weighted DiceCE loss for class imbalance
loss_fn = DiceCELoss(
    include_background=False,
    to_onehot_y=False,
    sigmoid=True,
    squared_pred=True,
    # Weight small structures (psoas) more
    ce_weight=torch.tensor([2.0, 2.0, 1.0, 1.0])  # [psoas_l, psoas_r, vat, inner]
)
```

**Action Items:**
- [ ] `step2_train_unet.py` → 4 output channels
- [ ] Dataset → `inner_abdomen.npy` loading ekle
- [ ] Augmentation pipeline güçlendir
- [ ] Loss weights ayarla
- [ ] (Optional) Vertebra heatmap/L3 locator ekle

---

## 🚀 Phase 3: Online Inference Pipeline (TO BE CREATED 🆕)

### Goal: Yeni vakalarda tam otomatik, hızlı VFA/PMA

### 3.1 Create Unified Inference Script

**File:** `inference_fast.py`

```python
#!/usr/bin/env python3
"""
Fast Online Inference Pipeline
-------------------------------
Input: CT DICOM or NIfTI
Process: Student model + HU thresholds
Output: VFA, PMA, PMI + QC + Overlay
"""

import torch
import numpy as np
from pathlib import Path
from monai.networks.nets import UNet
from core_mini import load_hu  # HU loading from DICOM/NIfTI

class FastInference:
    def __init__(self, model_path: str, device: str = "cuda"):
        """Initialize student model"""
        self.device = device
        self.model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=4,  # psoas_l, psoas_r, vat, inner_abdomen
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2
        ).to(device)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def find_l3_slice(self, ct_volume):
        """
        Find L3 slice automatically
        
        Options:
        1. Use vertebra heatmap from model
        2. Use HU-based vertebra detection (current core_mini method)
        3. Process multiple slices and select best
        """
        # Fallback: Use existing core_mini L3 selector
        from tools.l3_selector import find_l3_slice
        l3_idx = find_l3_slice(ct_volume)
        return l3_idx
    
    def segment_l3_slice(self, hu_slice: np.ndarray):
        """Run student model inference"""
        # Normalize HU
        hu_normalized = np.clip((hu_slice + 1000) / 2000.0, 0, 1).astype(np.float32)
        
        # Add batch and channel dims
        input_tensor = torch.from_numpy(hu_normalized[None, None, ...]).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            output = torch.sigmoid(output)  # [1, 4, H, W]
        
        # Threshold and convert to numpy
        masks = (output[0] > 0.5).cpu().numpy()
        psoas_left = masks[0]
        psoas_right = masks[1]
        vat = masks[2]
        inner_abdomen = masks[3]
        
        return {
            'psoas_left': psoas_left,
            'psoas_right': psoas_right,
            'vat': vat,
            'inner_abdomen': inner_abdomen
        }
    
    def calculate_areas(self, hu_slice, masks, pixel_spacing):
        """Calculate VFA and PMA with HU refinement"""
        pixel_area = pixel_spacing[0] * pixel_spacing[1]
        
        # PMA: Direct from masks
        psoas_mask = masks['psoas_left'] | masks['psoas_right']
        pma_mm2 = psoas_mask.sum() * pixel_area
        
        # VFA: Model mask ∩ HU fat range (refinement)
        fat_hu_mask = (hu_slice >= -190) & (hu_slice <= -30)
        inner_mask = masks['inner_abdomen']
        vfa_mask = fat_hu_mask & inner_mask
        vfa_mm2 = vfa_mask.sum() * pixel_area
        
        return {
            'pma_mm2': pma_mm2,
            'pma_cm2': pma_mm2 / 100,
            'vfa_mm2': vfa_mm2,
            'vfa_cm2': vfa_mm2 / 100
        }
    
    def quality_control(self, masks, areas):
        """Automated QC checks"""
        qc_flags = {
            'pass': True,
            'warnings': [],
            'errors': []
        }
        
        # Check 1: Inner abdomen size
        inner_area = masks['inner_abdomen'].sum()
        if inner_area < 5000:  # Too small
            qc_flags['errors'].append("Inner abdomen too small - possible segmentation failure")
            qc_flags['pass'] = False
        elif inner_area > 100000:  # Too large
            qc_flags['warnings'].append("Inner abdomen unusually large - check posterolateral leak")
        
        # Check 2: Psoas symmetry
        left_area = masks['psoas_left'].sum()
        right_area = masks['psoas_right'].sum()
        asymmetry = abs(left_area - right_area) / max(left_area, right_area, 1)
        if asymmetry > 0.5:
            qc_flags['warnings'].append(f"Psoas asymmetry {asymmetry:.1%} - check for pathology or segmentation error")
        
        # Check 3: VFA/PMA ratio
        if areas['vfa_mm2'] > 0 and areas['pma_mm2'] > 0:
            ratio = areas['vfa_mm2'] / areas['pma_mm2']
            if ratio < 0.5 or ratio > 20:
                qc_flags['warnings'].append(f"Unusual VFA/PMA ratio {ratio:.1f}")
        
        # Check 4: Zero areas
        if areas['vfa_mm2'] == 0:
            qc_flags['errors'].append("VFA is zero - check fat HU threshold or inner abdomen mask")
            qc_flags['pass'] = False
        if areas['pma_mm2'] == 0:
            qc_flags['errors'].append("PMA is zero - check psoas segmentation")
            qc_flags['pass'] = False
        
        return qc_flags
    
    def create_overlay(self, hu_slice, masks):
        """Create professional overlay visualization"""
        from core_mini import build_overlay
        
        # Use existing overlay function
        overlay = build_overlay(
            hu_slice,
            masks['inner_abdomen'],
            masks['vertebra'] if 'vertebra' in masks else None,
            masks['vat'],
            masks['psoas_left'] | masks['psoas_right']
        )
        
        return overlay
    
    def process_case(self, ct_path: str, output_dir: str, patient_height_cm: float):
        """Full pipeline for one case"""
        # Load CT
        ct_volume, spacing = load_ct(ct_path)
        
        # Find L3
        l3_idx = self.find_l3_slice(ct_volume)
        hu_slice = ct_volume[:, :, l3_idx]
        
        # Segment
        masks = self.segment_l3_slice(hu_slice)
        
        # Calculate areas
        areas = self.calculate_areas(hu_slice, masks, spacing[:2])
        
        # QC
        qc = self.quality_control(masks, areas)
        
        # Calculate PMI
        if patient_height_cm > 0:
            areas['pmi'] = areas['pma_cm2'] / (patient_height_cm / 100) ** 2
        
        # Create overlay
        overlay = self.create_overlay(hu_slice, masks)
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # JSON results
        results = {
            'case_id': Path(ct_path).stem,
            'l3_slice_index': l3_idx,
            'measurements': areas,
            'qc': qc,
            'pixel_spacing': spacing.tolist()
        }
        
        import json
        with open(output_path / f"{results['case_id']}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save overlay
        import cv2
        cv2.imwrite(str(output_path / f"{results['case_id']}_overlay.png"), overlay)
        
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--input", required=True, help="CT DICOM or NIfTI file")
    parser.add_argument("--output", default="./inference_output", help="Output directory")
    parser.add_argument("--height", type=float, default=0, help="Patient height (cm) for PMI")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    
    args = parser.parse_args()
    
    # Run inference
    inference = FastInference(args.model, device=args.device)
    results = inference.process_case(args.input, args.output, args.height)
    
    print("\n✅ Inference Complete!")
    print(f"📊 PMA: {results['measurements']['pma_cm2']:.1f} cm²")
    print(f"📊 VFA: {results['measurements']['vfa_cm2']:.1f} cm²")
    if 'pmi' in results['measurements']:
        print(f"📊 PMI: {results['measurements']['pmi']:.2f} cm²/m²")
    print(f"✅ QC: {'PASS' if results['qc']['pass'] else 'FAIL'}")
    if results['qc']['warnings']:
        print("⚠️  Warnings:", results['qc']['warnings'])
    if results['qc']['errors']:
        print("❌ Errors:", results['qc']['errors'])


if __name__ == "__main__":
    main()
```

**Action Items:**
- [ ] Create `inference_fast.py` script
- [ ] Integrate with `core_mini.py` utilities
- [ ] Test on sample DICOM files
- [ ] Benchmark speed (target: <5 seconds per case)

---

## 📦 Phase 4: Integration & Deployment

### 4.1 Unified CLI Interface

**File:** `l3_vfa_pma_cli.py`

```bash
# Single case
python l3_vfa_pma_cli.py --input case.dcm --height 170 --output results/

# Batch processing
python l3_vfa_pma_cli.py --input-dir /path/to/cases/ --output-dir results/ --batch

# With ground truth comparison
python l3_vfa_pma_cli.py --input case.dcm --gt-vfa 12000 --gt-pma 1500
```

### 4.2 GUI Integration

Update existing GUI to use fast inference:

```python
# desktop_project/l3_vfa_pma_gui.py
from inference_fast import FastInference

class L3GUI:
    def __init__(self):
        # Load student model
        self.inference = FastInference("models/student_model.pth")
    
    def process_dicom(self, dcm_path):
        # Use fast inference instead of core_mini
        results = self.inference.process_case(dcm_path, ...)
        self.display_results(results)
```

### 4.3 Model Versioning

```
models/
  ├── v1.0_baseline/
  │   ├── student_model.pth
  │   ├── training_config.json
  │   └── metrics.csv
  ├── v1.1_improved_fascia/
  │   └── student_model.pth
  └── latest -> v1.1_improved_fascia/
```

---

## 🧪 Phase 5: Validation & Benchmarking

### 5.1 Accuracy Validation

Compare student model vs ground truth:

```bash
# Run validation on AMOS22 test set
python validate_student.py \
  --model models/latest/student_model.pth \
  --test-cases data/amos22/test_cases.txt \
  --ground-truth data/ground_truth.csv \
  --output validation_report.json
```

**Metrics:**
- MAE (Mean Absolute Error): PMA, VFA
- Pearson correlation with ground truth
- Dice score: Psoas masks, Inner abdomen masks
- Processing time per case

### 5.2 Speed Benchmark

```bash
# Benchmark inference speed
python benchmark_speed.py \
  --model models/latest/student_model.pth \
  --test-cases 50 \
  --device cuda

# Expected results:
# - GPU: < 2 seconds per case
# - CPU: < 10 seconds per case
```

### 5.3 QC Flag Analysis

```bash
# Analyze QC flags on large dataset
python analyze_qc.py \
  --results-dir batch_results/ \
  --report qc_analysis.html

# Output:
# - QC pass rate: 95%
# - Common warnings: posterolateral leak (3%), psoas asymmetry (2%)
# - Error rate: 5%
```

---

## 📊 Success Criteria

### Phase 1 (Teacher Generation):
- ✅ All AMOS22 cases processed
- ✅ `inner_abdomen.npy` masks generated
- ✅ QC metadata for each case
- ✅ Leak detection flags

### Phase 2 (Training):
- ✅ Student model trained (60-100 epochs)
- ✅ Validation Dice > 0.85 (psoas), > 0.90 (inner abdomen)
- ✅ Model size < 50 MB
- ✅ Training metrics logged

### Phase 3 (Inference):
- ✅ `inference_fast.py` working
- ✅ Speed: < 5 seconds per case (GPU)
- ✅ QC automated
- ✅ Overlay quality comparable to comp2comp

### Phase 4 (Integration):
- ✅ CLI tool ready
- ✅ GUI updated
- ✅ Model versioning system

### Phase 5 (Validation):
- ✅ MAE < 50 mm² (VFA), < 15 mm² (PMA)
- ✅ Correlation > 0.90 with ground truth
- ✅ QC pass rate > 90%

---

## 🚀 Quick Start (After Implementation)

```bash
# 1. Download trained model
wget https://your-storage/models/student_model_v1.0.pth -O models/student_model.pth

# 2. Process a new case
python inference_fast.py \
  --model models/student_model.pth \
  --input new_patient.dcm \
  --height 175 \
  --output results/

# 3. View results
cat results/new_patient_results.json
open results/new_patient_overlay.png
```

---

## 📝 Next Steps

### Immediate (Week 1-2):
1. [ ] Phase 1.1: Add `inner_abdomen.npy` to teacher generation
2. [ ] Phase 1.2: Improve VAT mask quality
3. [ ] Phase 2.1: Update U-Net to 4 output channels
4. [ ] Phase 2.4: Tune loss function weights

### Short-term (Week 3-4):
5. [ ] Phase 2: Complete training (60-100 epochs)
6. [ ] Phase 3.1: Create `inference_fast.py`
7. [ ] Phase 5.1: Validation on test set

### Long-term (Month 2):
8. [ ] Phase 4: CLI/GUI integration
9. [ ] Phase 5.2-5.3: Benchmarking and QC analysis
10. [ ] Documentation and deployment

---

## 🔗 Related Files

- Teacher generation: `.github/azure-ml/step1_real_teachers_all.py`
- Training: `.github/azure-ml/step2_train_unet.py`
- Core utilities: `core_mini.py`
- Comp2Comp reference: `comp2comp_enhanced_l3.py`
- Current GUI: `desktop_project/l3_vfa_pma_gui.py`

---

**Last Updated:** 23 Aralık 2025
**Status:** Planning → Implementation Ready
