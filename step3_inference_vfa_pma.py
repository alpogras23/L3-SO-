#!/usr/bin/env python3
"""
Step 3: VFA/PMA Inference Pipeline
Runs inference on all AMOS cases using trained U-Net from Step 2
and calculates tissue areas using HU-based classification.

Usage:
    python step3_inference_vfa_pma.py \
        --teacher-labels-dir /path/to/teacher_labels \
        --checkpoint /path/to/step2_outputs/checkpoint.pth \
        --output-dir ./inference_results \
        --device cuda
"""

import json
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from monai.networks.nets import UNet
from tqdm import tqdm

# Import HU classifier
from hu_classifier import HUAnalyzer, HUThresholds


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class L3SliceDataset(Dataset):
    """Dataset for loading L3 slices from teacher labels."""
    
    def __init__(self, teacher_labels_dir: str):
        """
        Initialize dataset.
        
        Args:
            teacher_labels_dir: Path to teacher_labels directory
        """
        self.teacher_labels_dir = Path(teacher_labels_dir)
        self.case_dirs = sorted([d for d in self.teacher_labels_dir.iterdir() if d.is_dir()])
        
        if len(self.case_dirs) == 0:
            raise ValueError(f"No case directories found in {teacher_labels_dir}")
        
        logger.info(f"Found {len(self.case_dirs)} cases")
    
    def __len__(self) -> int:
        return len(self.case_dirs)
    
    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray, Dict]:
        """Load case data."""
        case_dir = self.case_dirs[idx]
        case_id = case_dir.name
        
        # Load HU slice
        hu_path = case_dir / "hu_slice.png"
        hu_slice = cv2.imread(str(hu_path), cv2.IMREAD_GRAYSCALE)
        
        # Convert from display PNG (0-255) back to HU values
        # Saved in stage1 as windowed [-150, 250] -> [0, 255]
        hu_slice = hu_slice.astype(np.float32)
        hu_slice = (hu_slice / 255.0) * 400.0 - 150.0
        
        # Load metadata
        meta_path = case_dir / "metadata.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        
        return case_id, hu_slice, metadata
    
    def get_case_path(self, idx: int) -> Path:
        """Get case directory path."""
        return self.case_dirs[idx]


class VFAPMAInference:
    """VFA/PMA inference pipeline."""
    
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 hu_thresholds: Optional[HUThresholds] = None,
                 pixel_spacing_mm: float = 0.977):
        """
        Initialize inference pipeline.
        
        Args:
            model: Trained U-Net model
            device: torch.device (cuda or cpu)
            hu_thresholds: HU threshold configuration
            pixel_spacing_mm: Pixel spacing in mm
        """
        self.model = model.to(device)
        self.device = device
        self.hu_thresholds = hu_thresholds or HUThresholds()
        self.analyzer = HUAnalyzer(
            thresholds=self.hu_thresholds,
            pixel_spacing_x=pixel_spacing_mm,
            pixel_spacing_y=pixel_spacing_mm
        )
        self.model.eval()
        
        logger.info(f"Model loaded on device: {device}")
        logger.info(f"HU Thresholds: Muscle [{self.hu_thresholds.MUSCLE_HU_MIN}, "
                   f"{self.hu_thresholds.MUSCLE_HU_MAX}], "
                   f"VAT [{self.hu_thresholds.VAT_HU_MIN}, "
                   f"{self.hu_thresholds.VAT_HU_MAX}]")
    
    def infer_single(self, hu_slice: np.ndarray) -> Dict:
        """
        Run inference on single slice.
        
        Args:
            hu_slice: 2D HU array
            
        Returns:
            Dictionary with masks and analysis results
        """
        # Prepare input
        hu_tensor = torch.tensor(hu_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        hu_tensor = hu_tensor.to(self.device)
        
        # Normalize HU to [0, 1] for the -150 to 250 range (same as training)
        # HU values from PNG are already clipped to -150 to 250 range
        hu_normalized = torch.clamp((hu_tensor + 150) / 400.0, 0, 1)
        
        # Infer
        with torch.no_grad():
            logits = self.model(hu_normalized)
            probs = F.softmax(logits, dim=1)
            segmentation = probs.argmax(dim=1)
        
        # Extract masks
        seg_numpy = segmentation.cpu().numpy()[0]
        
        # Create tissue masks
        # Class 0: background, 1: visceral (peritoneal), 2: subcutaneous
        peritoneal_mask = (seg_numpy == 1).astype(np.uint8)
        
        # Analyze with HU thresholds
        results = self.analyzer.analyze_slice(hu_slice, peritoneal_mask=peritoneal_mask)
        
        # Add segmentation masks
        results['segmentation'] = seg_numpy
        results['peritoneal_mask'] = peritoneal_mask
        
        return results
    
    def validate_measurements(self, pma_mm2: float, vat_mm2: float, 
                             sat_mm2: float) -> Tuple[bool, str]:
        """
        Validate tissue area measurements.
        
        Returns:
            (is_valid, message)
        """
        warnings = []
        
        # PMA range: 500-3000 mm²
        if not (500 <= pma_mm2 <= 3000):
            warnings.append(f"PMA {pma_mm2:.0f} outside typical range [500-3000]")
        
        # VAT range: 2000-100000 mm²
        if not (2000 <= vat_mm2 <= 100000):
            warnings.append(f"VAT {vat_mm2:.0f} outside typical range [2000-100000]")
        
        # SAT range: 1000-150000 mm²
        if not (1000 <= sat_mm2 <= 150000):
            warnings.append(f"SAT {sat_mm2:.0f} outside typical range [1000-150000]")
        
        # PMA should be much smaller than VAT
        if pma_mm2 > vat_mm2:
            warnings.append(f"Unusual: PMA ({pma_mm2:.0f}) > VAT ({vat_mm2:.0f})")
        
        is_valid = len(warnings) == 0
        message = "; ".join(warnings) if warnings else "OK"
        
        return is_valid, message


def create_overlay(hu_slice: np.ndarray, peritoneal_mask: np.ndarray, 
                  psoas_left: np.ndarray = None, psoas_right: np.ndarray = None) -> np.ndarray:
    """
    Create high-quality overlay image for QA.
    
    Args:
        hu_slice: HU values (H, W)
        peritoneal_mask: Visceral fat mask (H, W)
        psoas_left: Left psoas mask (H, W) - optional
        psoas_right: Right psoas mask (H, W) - optional
        
    Returns:
        RGB overlay image (H, W, 3)
    """
    # Normalize HU for display (-150 to 250 HU range)
    hu_norm = np.clip((hu_slice + 150) / 400 * 255, 0, 255).astype(np.uint8)
    
    # Create RGB base
    overlay = cv2.cvtColor(hu_norm, cv2.COLOR_GRAY2BGR)
    
    # Apply masks with colors
    # VFA (Visceral Fat Area) - Red
    if peritoneal_mask is not None:
        overlay[peritoneal_mask > 0] = [0, 0, 255]  # Red
    
    # PMA (Psoas Muscle Area) - Blue  
    if psoas_left is not None:
        overlay[psoas_left > 0] = [255, 0, 0]  # Blue
        
    if psoas_right is not None:
        overlay[psoas_right > 0] = [255, 0, 0]  # Blue
    
    return overlay


def load_checkpoint(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load trained U-Net from checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Create model
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=3,  # background, visceral, subcutaneous
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2
    )
    
    # Load state
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        loss = checkpoint.get('loss', 'unknown')
        logger.info(f"Loaded checkpoint from epoch {epoch}, loss: {loss}")
    else:
        # Assume it's just the state dict
        model.load_state_dict(checkpoint)
        logger.info("Loaded checkpoint")
    
    return model.to(device)


def process_batch(inference_pipeline: VFAPMAInference,
                  dataset: L3SliceDataset,
                  indices: list,
                  output_dir: Path) -> list:
    """Process a batch of cases."""
    results_list = []
    
    for idx in indices:
        case_id, hu_slice, metadata = dataset[idx]
        
        try:
            # Run inference
            results = inference_pipeline.infer_single(hu_slice)
            
            # Validate
            is_valid, msg = inference_pipeline.validate_measurements(
                results['pma_mm2'],
                results['vat_mm2'],
                results['sat_mm2']
            )
            
            # Prepare output
            output_data = {
                'case_id': case_id,
                'pma_mm2': float(results['pma_mm2']),
                'vat_mm2': float(results['vat_mm2']),
                'sat_mm2': float(results['sat_mm2']),
                'total_fat_mm2': float(results['total_fat_mm2']),
                'validity': {
                    'is_valid': is_valid,
                    'message': msg
                },
                'metadata': metadata,
                'hu_stats': inference_pipeline.analyzer.get_tissue_statistics(hu_slice)
            }
            
            # Save individual result
            output_file = output_dir / f"{case_id}_vfa_pma.json"
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            results_list.append(output_data)
            
            status = "✅" if is_valid else "⚠️"
            logger.info(f"{status} {case_id}: PMA={results['pma_mm2']:.0f}, "
                       f"VAT={results['vat_mm2']:.0f}, SAT={results['sat_mm2']:.0f}")
        
        except Exception as e:
            logger.error(f"❌ Error processing {case_id}: {e}")
            results_list.append({
                'case_id': case_id,
                'error': str(e)
            })
    
    return results_list


def main():
    """Main inference pipeline."""
    parser = argparse.ArgumentParser(description="VFA/PMA Inference Pipeline")
    parser.add_argument('--amos-data', type=str, required=True,
                       help='Path to AMOS22 data directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained U-Net checkpoint')
    parser.add_argument('--output-dir', type=str, default='./inference_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cuda', 'cpu', 'auto'],
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--pixel-spacing-mm', type=float, default=0.977,
                       help='Pixel spacing in mm')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip measurement validation')
    
    args = parser.parse_args()
    
    # Debug logging
    logger.info("Input arguments:")
    logger.info(f"  amos_data: {args.amos_data}")
    logger.info(f"  checkpoint: {args.checkpoint}")
    logger.info(f"  output_dir: {args.output_dir}")
    logger.info(f"  device: {args.device}")
    logger.info(f"  batch_size: {args.batch_size}")
    logger.info(f"  pixel_spacing_mm: {args.pixel_spacing_mm}")
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading AMOS data from {args.amos_data}")
    
    # Generate teacher labels on-the-fly using subprocess
    logger.info("Generating teacher labels...")
    teacher_labels_dir = Path(args.output_dir) / "teacher_labels_temp"
    teacher_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Run teacher generation as subprocess
    import subprocess
    cmd = [
        sys.executable, "step1_two_stage_l3_psoas.py",
        "--input-data", args.amos_data,
        "--output-dir", str(teacher_labels_dir)
    ]
    
    logger.info(f"Teacher generation command: {' '.join(cmd)}")
    logger.info(f"Input data path: {args.amos_data}")
    logger.info(f"Output dir: {teacher_labels_dir}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Teacher generation failed: {result.stderr}")
        raise RuntimeError(f"Teacher generation failed with code {result.returncode}")
    
    logger.info(f"Teacher generation completed: {result.stdout}")
    
    # Count successful cases
    success_count = len(list(teacher_labels_dir.glob("*/")))
    
    logger.info(f"Generated {success_count} teacher labels")
    
    dataset = L3SliceDataset(str(teacher_labels_dir))
    
    # Load checkpoint
    model = load_checkpoint(args.checkpoint, device)
    
    # Process all cases
    logger.info(f"Processing {len(dataset)} cases...")
    all_results = []
    
    for i in tqdm(range(len(dataset)), desc="Inference"):
        case_id, hu_slice, metadata = dataset[i]
        
        try:
            # Get pixel spacing from metadata, fallback to default
            pixel_area_mm2 = metadata.get('pixel_area_mm2', args.pixel_spacing_mm ** 2)
            pixel_spacing = np.sqrt(pixel_area_mm2)
            
            # Create case-specific inference pipeline
            case_inference = VFAPMAInference(model, device, pixel_spacing_mm=pixel_spacing)
            
            results = case_inference.infer_single(hu_slice)
            
            # Validate
            is_valid, msg = case_inference.validate_measurements(
                results['pma_mm2'],
                results['vat_mm2'],
                results['sat_mm2']
            )
            
            if not args.skip_validation and not is_valid:
                logger.warning(f"{case_id}: {msg}")
            
            # Save result
            output_data = {
                'case_id': case_id,
                'pma_mm2': float(results['pma_mm2']),
                'vat_mm2': float(results['vat_mm2']),
                'sat_mm2': float(results['sat_mm2']),
                'total_fat_mm2': float(results['total_fat_mm2']),
                'validity': {
                    'is_valid': is_valid,
                    'message': msg
                } if not args.skip_validation else None,
                'hu_stats': case_inference.analyzer.get_tissue_statistics(hu_slice),
                'metadata': metadata
            }
            
            output_file = output_dir / f"{case_id}_vfa_pma.json"
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            # Create and save overlay image
            overlay = create_overlay(
                hu_slice, 
                results['peritoneal_mask'],
                None,  # psoas_left not available from inference
                None   # psoas_right not available from inference
            )
            overlay_file = output_dir / f"{case_id}_overlay.png"
            cv2.imwrite(str(overlay_file), overlay)
            
            all_results.append(output_data)
        
        except Exception as e:
            logger.error(f"Error processing {case_id}: {e}")
            all_results.append({'case_id': case_id, 'error': str(e)})
    
    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'total_cases': len(dataset),
            'processed': len([r for r in all_results if 'pma_mm2' in r]),
            'errors': len([r for r in all_results if 'error' in r]),
            'results': all_results
        }, f, indent=2)
    
    logger.info("✅ Inference complete!")
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Summary: {summary_file}")
    
    # Print statistics
    valid_results = [r for r in all_results if 'pma_mm2' in r]
    if valid_results:
        pma_values = [r['pma_mm2'] for r in valid_results]
        vat_values = [r['vat_mm2'] for r in valid_results]
        
    logger.info("\nStatistics:")
    logger.info(f"PMA: {np.mean(pma_values):.0f} ± {np.std(pma_values):.0f} mm² "
               f"(range: {np.min(pma_values):.0f}-{np.max(pma_values):.0f})")
    logger.info(f"VAT: {np.mean(vat_values):.0f} ± {np.std(vat_values):.0f} mm² "
               f"(range: {np.min(vat_values):.0f}-{np.max(vat_values):.0f})")


if __name__ == "__main__":
    main()
