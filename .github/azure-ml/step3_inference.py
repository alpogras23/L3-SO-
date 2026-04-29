#!/usr/bin/env python3
"""
Azure ML Full Pipeline: Production Inference
Step 3: Use trained model for VFA/PMA calculation
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import zoom

# MONAI imports
from monai.networks.nets import UNet

def load_model(model_path, device):
    """Load trained U-Net model"""
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=3,  # psoas_left, psoas_right, vat
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def preprocess_hu_slice(hu_slice, target_size=256):
    """Preprocess HU slice for model"""
    # Normalize to [0, 1]
    hu_normalized = np.clip((hu_slice + 1000) / 2000.0, 0, 1).astype(np.float32)
    
    # Resize
    if hu_slice.shape[0] != target_size or hu_slice.shape[1] != target_size:
        scale = target_size / min(hu_slice.shape)
        hu_resized = zoom(hu_normalized, scale, order=1)
        
        # Center crop/pad
        h, w = hu_resized.shape
        if h > target_size:
            start = (h - target_size) // 2
            hu_resized = hu_resized[start:start+target_size, :]
        if w > target_size:
            start = (w - target_size) // 2
            hu_resized = hu_resized[:, start:start+target_size]
    else:
        hu_resized = hu_normalized
    
    # Add batch and channel dimensions [1, 1, H, W]
    return torch.from_numpy(hu_resized[np.newaxis, np.newaxis, ...])

def detect_l3_slice(ct_path):
    """Simple L3 detection: middle of volume"""
    # TODO: Implement proper L3 detection
    # For now: use middle slice
    img = nib.load(str(ct_path))
    vol = img.get_fdata()
    z_index = vol.shape[0] // 2
    return z_index

def calculate_vfa_pma(pred_masks, pixel_spacing):
    """
    Calculate VFA and PMA from predicted masks
    pred_masks: [3, H, W] - [psoas_left, psoas_right, vat]
    """
    pixel_area_mm2 = pixel_spacing[0] * pixel_spacing[1]
    
    # Extract masks
    psoas_left = pred_masks[0] > 0.5
    psoas_right = pred_masks[1] > 0.5
    vat = pred_masks[2] > 0.5
    
    # Calculate areas
    pma_mm2 = (psoas_left.sum() + psoas_right.sum()) * pixel_area_mm2
    vfa_mm2 = vat.sum() * pixel_area_mm2
    
    return float(vfa_mm2), float(pma_mm2)

def process_case(ct_path, model, device, img_size=256):
    """Process single case with model inference"""
    case_id = ct_path.stem
    print(f"Processing: {case_id}")
    
    try:
        # Load CT
        img = nib.load(str(ct_path))
        vol = img.get_fdata().astype(np.float32)
        spacing = img.header.get_zooms()
        
        # Detect L3 slice
        z_index = detect_l3_slice(ct_path)
        hu_slice = vol[z_index]
        
        print(f"  L3 slice: z={z_index}")
        print(f"  Slice shape: {hu_slice.shape}")
        print(f"  Pixel spacing: {spacing[:2]}")
        
        # Preprocess
        input_tensor = preprocess_hu_slice(hu_slice, target_size=img_size).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            pred_masks = torch.sigmoid(output[0]).cpu().numpy()  # [3, H, W]
        
        # Calculate VFA/PMA
        vfa_mm2, pma_mm2 = calculate_vfa_pma(pred_masks, spacing[:2])
        vfa_cm2 = vfa_mm2 / 100.0
        pma_cm2 = pma_mm2 / 100.0
        
        print(f"  VFA: {vfa_cm2:.2f} cm²")
        print(f"  PMA: {pma_cm2:.2f} cm²")
        
        result = {
            "case_id": case_id,
            "z_index": int(z_index),
            "vfa_mm2": float(vfa_mm2),
            "pma_mm2": float(pma_mm2),
            "vfa_cm2": float(vfa_cm2),
            "pma_cm2": float(pma_cm2),
            "pixel_spacing": [float(s) for s in spacing[:2]],
            "status": "success"
        }
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "case_id": case_id,
            "status": "error",
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='Production Inference with Trained Model')
    parser.add_argument('--input-data', required=True, help='Input AMOS22 NIfTI directory')
    parser.add_argument('--model-path', required=True, help='Path to trained model (.pth)')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--img-size', type=int, default=256, help='Model input size')
    args = parser.parse_args()
    
    input_dir = Path(args.input_data)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("STEP 3: Production Inference")
    print("=" * 70)
    print(f"Input: {input_dir}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, device)
    print("✓ Model loaded")
    print()
    
    # Find CT files
    ct_files = list(input_dir.glob("*_0000.nii.gz"))
    if not ct_files:
        ct_files = list(input_dir.glob("*.nii.gz"))
    
    print(f"Found {len(ct_files)} CT files")
    print()
    
    # Process all cases
    all_results = []
    for ct_path in ct_files:
        result = process_case(ct_path, model, device, args.img_size)
        all_results.append(result)
        
        # Save individual result
        result_path = output_dir / f"{result['case_id']}_results.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print()
    
    # Save summary
    summary = {
        "total_cases": len(all_results),
        "successful": sum(1 for r in all_results if r["status"] == "success"),
        "failed": sum(1 for r in all_results if r["status"] == "error"),
        "results": all_results
    }
    
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total: {summary['total_cases']}")
    print(f"Success: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Summary: {summary_path}")
    
    return 0 if summary['failed'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
