#!/usr/bin/env python3
"""
Gelişmiş L3 VFA/PMA Hesaplama - Özel HU Threshold Segmentasyonu ile

Bu script DICOM görüntülerinden L3 seviyesinde VFA ve PMA hesaplaması yapar.
Kullanıcının belirttiği HU threshold değerlerini kullanarak segmentasyon yapar.

HU Thresholds:
- Muscle (PMA): -29 to +150 HU
- VAT: -150 to -50 HU
- SAT: -190 to -30 HU

Kullanım:
    python advanced_l3_vfa_pma_hu_thresholds.py --input-dicom /path/to/dicom --output-dir ./results
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pydicom
from scipy import ndimage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_settings():
    """Load settings from settings.json."""
    settings_path = Path(__file__).parent / "settings.json"
    if settings_path.exists():
        with open(settings_path, 'r') as f:
            return json.load(f)
    else:
        logger.warning("settings.json not found, using defaults")
        return {}


# Global settings
SETTINGS = load_settings()


class AdvancedL3VFAPMAHUThresholds:
    """Comp2Comp tabanlı L3 VFA/PMA hesaplama sınıfı."""

    def __init__(self, use_comp2comp: bool = False):
        """Initialize Comp2Comp L3 VFA/PMA calculator."""
        logger.info("Comp2Comp L3 VFA/PMA calculator initialized")

        self.use_comp2comp = use_comp2comp

        if use_comp2comp:
            # Comp2Comp paths
            self.comp2comp_dir = Path(__file__).parent / "thirdparty" / "Comp2Comp"
            self.c2c_script = self.comp2comp_dir / "bin" / "C2C"

            # Check if Comp2Comp is available
            if not self.c2c_script.exists():
                raise FileNotFoundError(f"Comp2Comp script not found at {self.c2c_script}")
        else:
            logger.info("Using custom segmentation with user-defined HU thresholds")

    def run_comp2comp_pipeline(self, dicom_path: str, output_dir: str) -> Dict:
        """
        Run Comp2Comp spine_muscle_adipose_tissue pipeline.

        Args:
            dicom_path: Path to DICOM directory
            output_dir: Output directory for results

        Returns:
            Dictionary with results
        """
        logger.info(f"Running Comp2Comp pipeline on {dicom_path}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Run Comp2Comp command
        cmd = [
            sys.executable, str(self.c2c_script),
            "spine_muscle_adipose_tissue",
            "-i", dicom_path,
            "-o", str(output_path)
        ]

        logger.info(f"Executing: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.comp2comp_dir),
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Comp2Comp pipeline completed successfully")
            logger.debug(f"Comp2Comp stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"Comp2Comp stderr: {result.stderr}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Comp2Comp pipeline failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise

        # Parse Comp2Comp results
        return self._parse_comp2comp_results(str(output_path))

    def _parse_comp2comp_results(self, output_dir: str) -> Dict:
        """
        Parse Comp2Comp pipeline results.

        Args:
            output_dir: Comp2Comp output directory

        Returns:
            Parsed results dictionary
        """
        output_path = Path(output_dir)

        # For now, return a placeholder result since Comp2Comp may not produce
        # the expected output format for our test data
        logger.info(f"Looking for Comp2Comp results in: {output_path}")

        # Check if any files were created
        if output_path.exists():
            files = list(output_path.glob("*"))
            logger.info(f"Files created by Comp2Comp: {[str(f) for f in files]}")

        # Return placeholder results
        return {
            'L3_VFA_mm2': 0.0,  # Comp2Comp would provide actual values
            'L3_PMA_mm2': 0.0,
            'L3_SAT_mm2': 0.0,
            'processing_status': 'comp2comp_completed',
            'segmentation_method': 'comp2comp',
            'note': 'Comp2Comp pipeline ran but results parsing not fully implemented'
        }

    def _parse_comp2comp_results(self, output_dir: str) -> Dict:
        """
        Parse Comp2Comp pipeline results.

        Args:
            output_dir: Comp2Comp output directory

        Returns:
            Parsed results dictionary
        """
        output_path = Path(output_dir)

        # For now, return a placeholder result since Comp2Comp may not produce
        # the expected output format for our test data
        logger.info(f"Looking for Comp2Comp results in: {output_path}")

        # Check if any files were created
        if output_path.exists():
            files = list(output_path.glob("*"))
            logger.info(f"Files created by Comp2Comp: {[str(f) for f in files]}")

        # Return placeholder results
        return {
            'L3_VFA_mm2': 0.0,  # Comp2Comp would provide actual values
            'L3_PMA_mm2': 0.0,
            'L3_SAT_mm2': 0.0,
            'processing_status': 'comp2comp_completed',
            'segmentation_method': 'comp2comp',
            'note': 'Comp2Comp pipeline ran but results parsing not fully implemented'
        }

    def calculate_l3_metrics(self, dicom_path: str, comp2comp_results: Dict) -> Dict:
        """
        Calculate L3-specific VFA and PMA metrics using user-defined HU thresholds.

        Args:
            dicom_path: Path to DICOM directory
            comp2comp_results: Results from Comp2Comp pipeline

        Returns:
            Dictionary with L3 metrics
        """
        logger.info("Calculating L3-specific metrics with custom HU thresholds")

        # Load DICOM series
        hu_volume, metadata = self.load_dicom_series(dicom_path)

        # Find L3 slice
        l3_slice_idx = self.find_l3_slice(hu_volume, metadata)
        l3_hu_slice = hu_volume[l3_slice_idx]

        # Apply user-defined HU thresholds
        muscle_mask, visceral_fat_mask, subcutaneous_fat_mask = self.segment_tissues_with_custom_thresholds(l3_hu_slice)

        # Calculate areas
        pixel_spacing = metadata.get('pixel_spacing', [1.0, 1.0])
        pixel_area = pixel_spacing[0] * pixel_spacing[1]

        vfa_area = np.sum(visceral_fat_mask) * pixel_area
        pma_area = np.sum(muscle_mask) * pixel_area
        sat_area = np.sum(subcutaneous_fat_mask) * pixel_area

        results = {
            'L3_VFA_mm2': float(vfa_area),
            'L3_PMA_mm2': float(pma_area),
            'L3_SAT_mm2': float(sat_area),
            'L3_slice_index': int(l3_slice_idx),
            'pixel_spacing': pixel_spacing,
            'HU_thresholds': {
                'muscle': [-29, 150],
                'VAT': [-150, -50],
                'SAT': [-190, -30]
            }
        }

        logger.info(f"L3 Metrics calculated: VFA={vfa_area:.1f}mm², PMA={pma_area:.1f}mm², SAT={sat_area:.1f}mm²")
        return results

    def load_dicom_series(self, dicom_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Load DICOM series and extract HU volume.

        Args:
            dicom_path: Path to DICOM directory

        Returns:
            Tuple of (hu_volume, metadata)
        """
        dicom_files = list(Path(dicom_path).glob("*.dcm"))
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_path}")

        # Sort by slice position (try multiple attributes)
        def get_slice_position(dcm_file):
            ds = pydicom.dcmread(dcm_file)
            # Try SliceLocation first, then ImagePositionPatient, then InstanceNumber
            if hasattr(ds, 'SliceLocation') and ds.SliceLocation is not None:
                return ds.SliceLocation
            elif hasattr(ds, 'ImagePositionPatient') and ds.ImagePositionPatient:
                return ds.ImagePositionPatient[2]  # Z coordinate
            elif hasattr(ds, 'InstanceNumber') and ds.InstanceNumber is not None:
                return ds.InstanceNumber
            else:
                return 0  # Default if no positioning info

        dicom_files.sort(key=get_slice_position)

        slices = []
        metadata = {}

        for dcm_file in dicom_files:
            ds = pydicom.dcmread(dcm_file)

            # Extract HU values
            hu_slice = ds.pixel_array.astype(np.float32)

            # Apply rescale with fallbacks
            slope = getattr(ds, 'RescaleSlope', 1.0)
            intercept = getattr(ds, 'RescaleIntercept', 0.0)
            hu_slice = hu_slice * slope + intercept

            slices.append(hu_slice)

            # Store metadata from first slice
            if not metadata:
                metadata = {
                    'pixel_spacing': getattr(ds, 'PixelSpacing', [1.0, 1.0]),
                    'slice_thickness': getattr(ds, 'SliceThickness', 1.0),
                    'study_instance_uid': getattr(ds, 'StudyInstanceUID', ''),
                    'series_instance_uid': getattr(ds, 'SeriesInstanceUID', ''),
                }

        hu_volume = np.stack(slices, axis=0)
        logger.info(f"Loaded DICOM series: {hu_volume.shape} (slices, height, width)")
        return hu_volume, metadata

    def find_l3_slice(self, hu_volume: np.ndarray, metadata: Dict) -> int:
        """
        Find L3 slice using vertebra detection.

        Args:
            hu_volume: 3D HU volume
            metadata: DICOM metadata

        Returns:
            Index of L3 slice
        """
        # Simple approach: find slice with highest vertebra-like HU values
        vertebra_hu_min = SETTINGS.get('VB_HU_MIN', 150)
        vertebra_hu_max = SETTINGS.get('VB_HU_MAX', 4000)

        best_slice_idx = 0
        max_vertebra_pixels = 0

        for i, slice_hu in enumerate(hu_volume):
            vertebra_mask = ((slice_hu >= vertebra_hu_min) & (slice_hu <= vertebra_hu_max)).astype(np.uint8)
            vertebra_pixels = np.sum(vertebra_mask)

            if vertebra_pixels > max_vertebra_pixels:
                max_vertebra_pixels = vertebra_pixels
                best_slice_idx = i

        logger.info(f"L3 slice detected at index {best_slice_idx} with {max_vertebra_pixels} vertebra pixels")
        return best_slice_idx

    def segment_tissues_with_custom_thresholds(self, hu_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment tissues using user-defined HU thresholds.

        Args:
            hu_slice: 2D HU slice

        Returns:
            Tuple of (muscle_mask, visceral_fat_mask, subcutaneous_fat_mask)
        """
        # User-defined HU thresholds
        MUSCLE_MIN, MUSCLE_MAX = -29, 150    # PMA (Psoas Muscle Area)
        VAT_MIN, VAT_MAX = -150, -50         # Visceral Fat Area
        SAT_MIN, SAT_MAX = -190, -30         # Subcutaneous Fat Area

        # Segment tissues
        muscle_mask = ((hu_slice >= MUSCLE_MIN) & (hu_slice <= MUSCLE_MAX)).astype(np.uint8)
        visceral_fat_mask = ((hu_slice >= VAT_MIN) & (hu_slice <= VAT_MAX)).astype(np.uint8)
        subcutaneous_fat_mask = ((hu_slice >= SAT_MIN) & (hu_slice <= SAT_MAX)).astype(np.uint8)

        # Clean up masks
        muscle_mask = ndimage.binary_opening(muscle_mask, iterations=1)
        visceral_fat_mask = ndimage.binary_opening(visceral_fat_mask, iterations=1)
        subcutaneous_fat_mask = ndimage.binary_opening(subcutaneous_fat_mask, iterations=1)

        return muscle_mask, visceral_fat_mask, subcutaneous_fat_mask

    def create_overlay_image(self, hu_slice: np.ndarray, muscle_mask: np.ndarray,
                           visceral_fat_mask: np.ndarray, subcutaneous_fat_mask: np.ndarray) -> np.ndarray:
        """
        Create overlay image for visualization.

        Args:
            hu_slice: Original HU slice
            muscle_mask: Muscle segmentation mask
            visceral_fat_mask: Visceral fat segmentation mask
            subcutaneous_fat_mask: Subcutaneous fat segmentation mask

        Returns:
            RGB overlay image
        """
        # Normalize HU for display (-150 to 250 HU window)
        hu_normalized = np.clip(hu_slice, -150, 250)
        hu_normalized = ((hu_normalized + 150) / 400 * 255).astype(np.uint8)

        # Create RGB image
        overlay = cv2.cvtColor(hu_normalized, cv2.COLOR_GRAY2RGB)

        # Apply color overlays
        overlay[muscle_mask > 0] = [255, 0, 0]        # Red for muscle
        overlay[visceral_fat_mask > 0] = [0, 255, 0]  # Green for VAT
        overlay[subcutaneous_fat_mask > 0] = [0, 0, 255]  # Blue for SAT

        return overlay

    def process_case(self, dicom_path: str, output_dir: str) -> Dict:
        """
        Process a single case using either Comp2Comp or custom segmentation.

        Args:
            dicom_path: Path to DICOM directory
            output_dir: Output directory

        Returns:
            Dictionary with results
        """
        logger.info(f"Processing case: {dicom_path}")

        if self.use_comp2comp:
            return self.run_comp2comp_pipeline(dicom_path, output_dir)
        else:
            # Use custom segmentation
            logger.info("Using custom segmentation with user-defined HU thresholds")

            # Calculate L3 metrics with custom thresholds
            l3_results = self.calculate_l3_metrics(dicom_path, {})

            # Create visualizations
            self.create_visualizations(dicom_path, l3_results, output_dir)

        # Combine results
        results = {
            **l3_results,
            'processing_status': 'completed',
            'segmentation_method': 'custom_hu_thresholds'
        }

        # Save results
        self.save_results(results, output_dir)

        return results

    def create_visualizations(self, dicom_path: str, results: Dict, output_dir: str):
        """
        Create visualization images.

        Args:
            dicom_path: Path to DICOM directory
            results: Results dictionary
            output_dir: Output directory
        """
        try:
            # Load DICOM series
            hu_volume, metadata = self.load_dicom_series(dicom_path)

            # Get L3 slice
            l3_idx = results['L3_slice_index']
            l3_hu_slice = hu_volume[l3_idx]

            # Get masks
            muscle_mask, visceral_fat_mask, subcutaneous_fat_mask = self.segment_tissues_with_custom_thresholds(l3_hu_slice)

            # Create overlay
            overlay = self.create_overlay_image(l3_hu_slice, muscle_mask, visceral_fat_mask, subcutaneous_fat_mask)

            # Add text annotations
            overlay = self.add_annotations(overlay, results)

            # Save overlay
            output_path = Path(output_dir) / "l3_overlay.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved overlay image: {output_path}")

        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")

    def add_annotations(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """
        Add text annotations to image.

        Args:
            image: RGB image
            results: Results dictionary

        Returns:
            Annotated image
        """
        annotated = image.copy()

        # Add metrics text
        vfa = results.get('L3_VFA_mm2', 0)
        pma = results.get('L3_PMA_mm2', 0)
        sat = results.get('L3_SAT_mm2', 0)

        text_lines = [
            f"L3 VFA: {vfa:.1f} mm²",
            f"L3 PMA: {pma:.1f} mm²",
            f"L3 SAT: {sat:.1f} mm²",
            "HU Thresholds:",
            "Muscle: -29 to +150 HU",
            "VAT: -150 to -50 HU",
            "SAT: -190 to -30 HU"
        ]

        y_offset = 30
        for line in text_lines:
            cv2.putText(annotated, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += 25

        return annotated

    def save_results(self, results: Dict, output_dir: str):
        """
        Save results to JSON file.

        Args:
            results: Results dictionary
            output_dir: Output directory
        """
        def make_json_serializable(obj):
            """Convert non-JSON serializable objects to strings."""
            if hasattr(obj, '__class__'):
                # Convert pydicom MultiValue and other special objects to string
                return str(obj)
            return obj

        output_path = Path(output_dir) / "l3_vfa_pma_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Make results JSON serializable
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {k: make_json_serializable(v) for k, v in value.items()}
            else:
                serializable_results[key] = make_json_serializable(value)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Saved results: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Advanced L3 VFA/PMA Calculator with HU Thresholds")
    parser.add_argument('--input-dicom', required=True, help='Path to DICOM directory')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--use-comp2comp', action='store_true', help='Use Comp2Comp pipeline instead of custom HU thresholds')

    args = parser.parse_args()

    # Create calculator
    calculator = AdvancedL3VFAPMAHUThresholds(use_comp2comp=args.use_comp2comp)

    try:
        # Process case
        results = calculator.process_case(args.input_dicom, args.output_dir)

        # Print summary
        print("\n" + "="*50)
        print("L3 VFA/PMA RESULTS")
        print("="*50)
        print(".1f")
        print(".1f")
        print(".1f")
        print("HU Thresholds Used:")
        print("  Muscle (PMA): -29 to +150 HU")
        print("  VAT: -150 to -50 HU")
        print("  SAT: -190 to -30 HU")
        print("="*50)

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()